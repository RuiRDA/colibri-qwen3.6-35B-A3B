/* Colibri CPU engine for Qwen3.6-35B-A3B (Qwen3_5Moe text architecture).
 *
 * Text-only, sequential inference.  The 30 Gated DeltaNet layers keep recurrent
 * FP32 state; the 10 full-attention layers keep a bounded KV cache.  Routed
 * expert weights remain on disk and are loaded into a small per-layer LRU.
 * Dense and shared weights use the converter's row-wise int4 container, while
 * embeddings/lm_head default to int8.
 *
 * Build:   make -C c qwen36
 * Run:     SNAP=/path/to/model PROMPT='<|im_start|>user\nHello...' ./c/qwen36 4
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#if defined(__APPLE__) || defined(__linux__)
#include <sys/resource.h>
#endif
#include "compat.h"
#include "json.h"
#include "st.h"
#include "tok.h"
#ifdef _OPENMP
#include <omp.h>
#else
static inline int omp_get_max_threads(void){ return 1; }
#endif
#ifdef __AVX2__
#include <immintrin.h>
static inline float hsum256(__m256 v){
    __m128 lo=_mm256_castps256_ps128(v), hi=_mm256_extractf128_ps(v,1);
    lo=_mm_add_ps(lo,hi); __m128 sh=_mm_movehl_ps(lo,lo); lo=_mm_add_ps(lo,sh);
    sh=_mm_shuffle_ps(lo,lo,1); lo=_mm_add_ss(lo,sh); return _mm_cvtss_f32(lo);
}
#endif

typedef struct {
    int hidden,layers,vocab,experts,topk,moe_inter,shared_inter;
    int q_heads,kv_heads,head_dim,rotary_dim,full_interval;
    int lk_heads,lv_heads,lk_dim,lv_dim,conv_kernel;
    int eos; float eps,theta;
} Cfg;

typedef struct { int fmt,O,I; float *f,*s; int8_t *q8; uint8_t *q4; } QT;

typedef struct {
    float *in_norm,*post_norm;
    int full;
    QT q,k,v,o; float *q_norm,*k_norm;
    QT qkv,z,b,a,out; float *conv,*A_log,*dt_bias,*lin_norm;
    float *conv_state,*rec_state;
    float *router;
    QT sh_gate,sh_up,sh_down,sh_control;
} Layer;

typedef struct { int eid; uint64_t used; QT gate,up,down; } ESlot;

typedef struct {
    Cfg c; shards shards; Layer *L;
    QT embed,lm_head; float *final_norm;
    ESlot **cache; int *cache_n,cache_cap; uint64_t clock,hits,misses;
    float **K,**V; int max_ctx;
    int64_t resident_bytes;
} Model;

static float g_temp=0.6f, g_topp=0.95f;
static int g_topk=0, g_ngen=512;
static uint64_t g_rng=0x9E3779B97F4A7C15ULL;

static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static double rss_gb(void){ struct rusage r; getrusage(RUSAGE_SELF,&r);
#ifdef __APPLE__
    return r.ru_maxrss/(1024.0*1024.0*1024.0);
#else
    return r.ru_maxrss/(1024.0*1024.0);
#endif
}
static void *xmalloc(size_t n){ void *p=malloc(n?n:1); if(!p){fprintf(stderr,"out of memory (%zu bytes)\n",n);exit(1);} return p; }
static float *falloc(int64_t n){ if(n<0||(uint64_t)n>SIZE_MAX/4){fprintf(stderr,"invalid allocation\n");exit(1);} return xmalloc((size_t)n*4); }
static float sigmoidf1(float x){ return 1.f/(1.f+expf(-x)); }
static float siluf1(float x){ return x*sigmoidf1(x); }
static float softplusf1(float x){ return x>20.f?x:log1pf(expf(x)); }

static void softmax(float *x,int n){
    float mx=x[0]; for(int i=1;i<n;i++) if(x[i]>mx) mx=x[i];
    double s=0; for(int i=0;i<n;i++){ x[i]=expf(x[i]-mx); s+=x[i]; }
    for(int i=0;i<n;i++) x[i]/=(float)s;
}

static void rms_centered(float *out,const float *x,const float *w,int n,float eps){
    double ss=0; for(int i=0;i<n;i++) ss+=(double)x[i]*x[i];
    float r=1.f/sqrtf((float)(ss/n)+eps);
    for(int i=0;i<n;i++) out[i]=x[i]*r*(1.f+w[i]);
}
static void rms_gated(float *out,const float *x,const float *w,const float *gate,int n,float eps){
    double ss=0; for(int i=0;i<n;i++) ss+=(double)x[i]*x[i];
    float r=1.f/sqrtf((float)(ss/n)+eps);
    for(int i=0;i<n;i++) out[i]=x[i]*r*w[i]*siluf1(gate[i]);
}
static void l2_normalize(float *out,const float *x,int n){
    double ss=0; for(int i=0;i<n;i++) ss+=(double)x[i]*x[i];
    float r=1.f/sqrtf((float)ss+1e-6f); for(int i=0;i<n;i++) out[i]=x[i]*r;
}

static void matmul_f(float *y,const float *x,const float *w,int I,int O){
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){ const float *wr=w+(int64_t)o*I; float a=0; int i=0;
#ifdef __AVX2__
        __m256 acc=_mm256_setzero_ps();
        for(;i+8<=I;i+=8) acc=_mm256_fmadd_ps(_mm256_loadu_ps(x+i),_mm256_loadu_ps(wr+i),acc);
        a=hsum256(acc);
#endif
        for(;i<I;i++) a+=x[i]*wr[i]; y[o]=a;
    }
}
static void matmul_q8(float *y,const float *x,const int8_t *q,const float *sc,int I,int O){
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){ const int8_t *wr=q+(int64_t)o*I; float a=0; int i=0;
#ifdef __AVX2__
        __m256 acc=_mm256_setzero_ps();
        for(;i+8<=I;i+=8){
            __m256i wi=_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(wr+i)));
            acc=_mm256_fmadd_ps(_mm256_loadu_ps(x+i),_mm256_cvtepi32_ps(wi),acc);
        }
        a=hsum256(acc);
#endif
        for(;i<I;i++) a+=x[i]*(float)wr[i]; y[o]=a*sc[o];
    }
}
static void matmul_q4(float *y,const float *x,const uint8_t *q,const float *sc,int I,int O){
    int rb=(I+1)/2;
    #pragma omp parallel for schedule(static)
    for(int o=0;o<O;o++){ const uint8_t *wr=q+(int64_t)o*rb; float a=0; int i=0;
#ifdef __AVX2__
        const __m128i mask=_mm_set1_epi8(15); const __m256i bias=_mm256_set1_epi32(8);
        __m256 acc=_mm256_setzero_ps();
        for(;i+16<=I;i+=16){
            __m128i by=_mm_loadl_epi64((const __m128i*)(wr+(i>>1)));
            __m128i lo=_mm_and_si128(by,mask), hi=_mm_and_si128(_mm_srli_epi16(by,4),mask);
            __m128i nb=_mm_unpacklo_epi8(lo,hi);
            __m256 w0=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(nb),bias));
            __m256 w1=_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(nb,8)),bias));
            acc=_mm256_fmadd_ps(_mm256_loadu_ps(x+i),w0,acc);
            acc=_mm256_fmadd_ps(_mm256_loadu_ps(x+i+8),w1,acc);
        }
        a=hsum256(acc);
#endif
        for(;i<I;i++){ uint8_t b=wr[i>>1]; int v=((i&1)?b>>4:b&15)-8; a+=x[i]*(float)v; }
        y[o]=a*sc[o];
    }
}
static void matvec(float *y,const float *x,const QT *w){
    if(w->fmt==0) matmul_f(y,x,w->f,w->I,w->O);
    else if(w->fmt==1) matmul_q8(y,x,w->q8,w->s,w->I,w->O);
    else matmul_q4(y,x,w->q4,w->s,w->I,w->O);
}

static void qt_free(QT *t){ free(t->f); free(t->q8); free(t->q4); free(t->s); memset(t,0,sizeof(*t)); }
static int64_t qt_bytes(const QT *t){ int64_t n=(int64_t)t->O*t->I;
    return t->fmt==0?n*4:t->fmt==1?n+(int64_t)t->O*4:(int64_t)t->O*((t->I+1)/2)+(int64_t)t->O*4; }
static QT qt_load(Model *m,const char *name,int O,int I,int drop){
    QT t={0}; t.O=O; t.I=I; char sn[384]; snprintf(sn,sizeof(sn),"%s.qs",name);
    if(st_has(&m->shards,sn)){
        int64_t nb=st_nbytes(&m->shards,name), q8=(int64_t)O*I, q4=(int64_t)O*((I+1)/2);
        if(nb==q8){ t.fmt=1; t.q8=xmalloc((size_t)nb); st_read_raw(&m->shards,name,t.q8,drop); }
        else if(nb==q4){ t.fmt=2; t.q4=xmalloc((size_t)nb); st_read_raw(&m->shards,name,t.q4,drop); }
        else { fprintf(stderr,"bad quantized shape for %s (%lld bytes)\n",name,(long long)nb); exit(1); }
        t.s=falloc(O); st_read_f32(&m->shards,sn,t.s,drop);
    } else {
        if(st_numel(&m->shards,name)!=(int64_t)O*I){ fprintf(stderr,"missing/bad tensor %s\n",name); exit(1); }
        t.fmt=0; t.f=falloc((int64_t)O*I); st_read_f32(&m->shards,name,t.f,drop);
    }
    return t;
}
static float *load_f(Model *m,const char *name,int64_t expected){
    int64_t n=st_numel(&m->shards,name); if(n!=expected){fprintf(stderr,"missing/bad %s: got %lld expected %lld\n",name,(long long)n,(long long)expected);exit(1);}
    float *p=falloc(n); st_read_f32(&m->shards,name,p,0); return p;
}

static jval *config_root(const char *snap,char **arena){
    char path[2048]; snprintf(path,sizeof(path),"%s/config.json",snap); FILE *f=fopen(path,"rb");
    if(!f){perror(path);exit(1);} fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *b=xmalloc((size_t)n+1); if(fread(b,1,n,f)!=(size_t)n){fprintf(stderr,"short config read\n");exit(1);} fclose(f); b[n]=0;
    jval *r=json_parse(b,arena); free(b); return r;
}
static int gi(jval *r,const char *k){ jval *v=json_get(r,k); return v?(int)v->num:0; }
static float gf(jval *r,const char *k,float d){ jval *v=json_get(r,k); return v?(float)v->num:d; }
static void load_cfg(Cfg *c,const char *snap){
    char *arena=NULL; jval *outer=config_root(snap,&arena), *r=json_get(outer,"text_config"); if(!r) r=outer;
    c->hidden=gi(r,"hidden_size"); c->layers=gi(r,"num_hidden_layers"); c->vocab=gi(r,"vocab_size");
    c->experts=gi(r,"num_experts"); c->topk=gi(r,"num_experts_per_tok");
    c->moe_inter=gi(r,"moe_intermediate_size"); c->shared_inter=gi(r,"shared_expert_intermediate_size");
    c->q_heads=gi(r,"num_attention_heads"); c->kv_heads=gi(r,"num_key_value_heads"); c->head_dim=gi(r,"head_dim");
    c->full_interval=gi(r,"full_attention_interval"); c->lk_heads=gi(r,"linear_num_key_heads");
    c->lv_heads=gi(r,"linear_num_value_heads"); c->lk_dim=gi(r,"linear_key_head_dim");
    c->lv_dim=gi(r,"linear_value_head_dim"); c->conv_kernel=gi(r,"linear_conv_kernel_dim");
    c->eps=gf(r,"rms_norm_eps",1e-6f); c->eos=gi(r,"eos_token_id");
    float partial=gf(r,"partial_rotary_factor",1.f); c->rotary_dim=(int)(c->head_dim*partial);
    jval *rp=json_get(r,"rope_parameters"); c->theta=rp?gf(rp,"rope_theta",10000000.f):10000000.f;
    if(c->hidden<1||c->layers<1||c->experts<1||c->topk<1||c->q_heads<1||c->kv_heads<1||
       c->head_dim<1||c->lv_heads<1||c->lk_heads<1||c->q_heads%c->kv_heads||c->lv_heads%c->lk_heads){
        fprintf(stderr,"invalid or unsupported Qwen3.6 text_config\n"); exit(1);
    }
    if(c->rotary_dim&1){fprintf(stderr,"rotary dimension must be even\n");exit(1);} free(arena);
}

static void load_layer(Model *m,int li){
    Cfg *c=&m->c; Layer *l=&m->L[li]; char n[384];
#define NAME(s) (snprintf(n,sizeof(n),"model.layers.%d.%s",li,s),n)
    l->in_norm=load_f(m,NAME("input_layernorm.weight"),c->hidden);
    l->post_norm=load_f(m,NAME("post_attention_layernorm.weight"),c->hidden);
    l->full=c->full_interval>0 && (li+1)%c->full_interval==0;
    if(l->full){
        l->q=qt_load(m,NAME("self_attn.q_proj.weight"),c->q_heads*c->head_dim*2,c->hidden,0);
        l->k=qt_load(m,NAME("self_attn.k_proj.weight"),c->kv_heads*c->head_dim,c->hidden,0);
        l->v=qt_load(m,NAME("self_attn.v_proj.weight"),c->kv_heads*c->head_dim,c->hidden,0);
        l->o=qt_load(m,NAME("self_attn.o_proj.weight"),c->hidden,c->q_heads*c->head_dim,0);
        l->q_norm=load_f(m,NAME("self_attn.q_norm.weight"),c->head_dim);
        l->k_norm=load_f(m,NAME("self_attn.k_norm.weight"),c->head_dim);
    } else {
        int key=c->lk_heads*c->lk_dim, val=c->lv_heads*c->lv_dim, conv=2*key+val;
        l->qkv=qt_load(m,NAME("linear_attn.in_proj_qkv.weight"),conv,c->hidden,0);
        l->z=qt_load(m,NAME("linear_attn.in_proj_z.weight"),val,c->hidden,0);
        l->b=qt_load(m,NAME("linear_attn.in_proj_b.weight"),c->lv_heads,c->hidden,0);
        l->a=qt_load(m,NAME("linear_attn.in_proj_a.weight"),c->lv_heads,c->hidden,0);
        l->out=qt_load(m,NAME("linear_attn.out_proj.weight"),c->hidden,val,0);
        l->conv=load_f(m,NAME("linear_attn.conv1d.weight"),(int64_t)conv*c->conv_kernel);
        l->A_log=load_f(m,NAME("linear_attn.A_log"),c->lv_heads);
        l->dt_bias=load_f(m,NAME("linear_attn.dt_bias"),c->lv_heads);
        l->lin_norm=load_f(m,NAME("linear_attn.norm.weight"),c->lv_dim);
        l->conv_state=calloc((size_t)conv*c->conv_kernel,sizeof(float));
        l->rec_state=calloc((size_t)c->lv_heads*c->lk_dim*c->lv_dim,sizeof(float));
        if(!l->conv_state||!l->rec_state){fprintf(stderr,"out of memory for DeltaNet state\n");exit(1);}
    }
    l->router=load_f(m,NAME("mlp.gate.weight"),(int64_t)c->experts*c->hidden);
    l->sh_gate=qt_load(m,NAME("mlp.shared_expert.gate_proj.weight"),c->shared_inter,c->hidden,0);
    l->sh_up=qt_load(m,NAME("mlp.shared_expert.up_proj.weight"),c->shared_inter,c->hidden,0);
    l->sh_down=qt_load(m,NAME("mlp.shared_expert.down_proj.weight"),c->hidden,c->shared_inter,0);
    l->sh_control=qt_load(m,NAME("mlp.shared_expert_gate.weight"),1,c->hidden,0);
#undef NAME
    m->resident_bytes+=(int64_t)c->hidden*8;
    QT *all[]={&l->q,&l->k,&l->v,&l->o,&l->qkv,&l->z,&l->b,&l->a,&l->out,
              &l->sh_gate,&l->sh_up,&l->sh_down,&l->sh_control};
    for(size_t i=0;i<sizeof(all)/sizeof(all[0]);i++) if(all[i]->O) m->resident_bytes+=qt_bytes(all[i]);
    m->resident_bytes+=(int64_t)c->experts*c->hidden*4;
}

static void model_init(Model *m,const char *snap,int cap,int max_ctx){
    memset(m,0,sizeof(*m)); load_cfg(&m->c,snap); st_init(&m->shards,snap); Cfg *c=&m->c;
    m->cache_cap=cap<1?1:cap; m->max_ctx=max_ctx;
    m->embed=qt_load(m,"model.embed_tokens.weight",c->vocab,c->hidden,0);
    m->lm_head=qt_load(m,"lm_head.weight",c->vocab,c->hidden,0);
    m->final_norm=load_f(m,"model.norm.weight",c->hidden);
    m->resident_bytes=qt_bytes(&m->embed)+qt_bytes(&m->lm_head)+(int64_t)c->hidden*4;
    m->L=calloc((size_t)c->layers,sizeof(Layer)); m->cache=calloc((size_t)c->layers,sizeof(ESlot*));
    m->cache_n=calloc((size_t)c->layers,sizeof(int)); m->K=calloc((size_t)c->layers,sizeof(float*));
    m->V=calloc((size_t)c->layers,sizeof(float*)); if(!m->L||!m->cache||!m->cache_n||!m->K||!m->V){fprintf(stderr,"out of memory\n");exit(1);}
    for(int li=0;li<c->layers;li++){
        load_layer(m,li); m->cache[li]=calloc((size_t)m->cache_cap,sizeof(ESlot));
        if(!m->cache[li]){fprintf(stderr,"out of memory for expert cache\n");exit(1);}
        if(m->L[li].full){ int64_t n=(int64_t)max_ctx*c->kv_heads*c->head_dim;
            m->K[li]=calloc((size_t)n,sizeof(float)); m->V[li]=calloc((size_t)n,sizeof(float));
            if(!m->K[li]||!m->V[li]){fprintf(stderr,"out of memory for KV cache\n");exit(1);}
        }
        fprintf(stderr,"\rloading layer %d/%d",li+1,c->layers); fflush(stderr);
    }
    fprintf(stderr,"\n");
}

static void model_reset(Model *m){ Cfg *c=&m->c;
    for(int li=0;li<c->layers;li++) if(!m->L[li].full){
        int conv=2*c->lk_heads*c->lk_dim+c->lv_heads*c->lv_dim;
        memset(m->L[li].conv_state,0,(size_t)conv*c->conv_kernel*4);
        memset(m->L[li].rec_state,0,(size_t)c->lv_heads*c->lk_dim*c->lv_dim*4);
    }
}

static void embed_row(Model *m,int token,float *x){ QT *e=&m->embed; int D=m->c.hidden;
    if(token<0||token>=m->c.vocab){fprintf(stderr,"token %d out of range\n",token);exit(1);}
    if(e->fmt==0) memcpy(x,e->f+(int64_t)token*D,(size_t)D*4);
    else if(e->fmt==1){ const int8_t *q=e->q8+(int64_t)token*D; float s=e->s[token]; for(int i=0;i<D;i++) x[i]=q[i]*s; }
    else { const uint8_t *q=e->q4+(int64_t)token*((D+1)/2); float s=e->s[token];
        for(int i=0;i<D;i++){uint8_t b=q[i>>1];x[i]=(((i&1)?b>>4:b&15)-8)*s;} }
}

static void rope_partial(float *x,int pos,int rd,float theta){
    int half=rd/2; float tmp[512]; if(rd>(int)(sizeof(tmp)/sizeof(tmp[0]))){fprintf(stderr,"rope dimension too large\n");exit(1);}
    memcpy(tmp,x,(size_t)rd*4); for(int j=0;j<half;j++){
        float angle=pos*powf(theta,-2.f*j/rd), cs=cosf(angle), sn=sinf(angle);
        x[j]=tmp[j]*cs-tmp[j+half]*sn; x[j+half]=tmp[j+half]*cs+tmp[j]*sn;
    }
}

static void full_attention(Model *m,Layer *l,int li,const float *x,int pos,float *out){
    Cfg *c=&m->c; int QH=c->q_heads,KH=c->kv_heads,HD=c->head_dim,QD=QH*HD;
    float *qr=falloc(2*QD),*kr=falloc(KH*HD),*vr=falloc(KH*HD),*q=falloc(QD),*ctx=falloc(QD),*scores=falloc(pos+1);
    matvec(qr,x,&l->q); matvec(kr,x,&l->k); matvec(vr,x,&l->v);
    for(int h=0;h<QH;h++){ rms_centered(q+h*HD,qr+h*2*HD,l->q_norm,HD,c->eps); rope_partial(q+h*HD,pos,c->rotary_dim,c->theta); }
    for(int h=0;h<KH;h++){ float *dst=m->K[li]+((int64_t)pos*KH+h)*HD;
        rms_centered(dst,kr+h*HD,l->k_norm,HD,c->eps); rope_partial(dst,pos,c->rotary_dim,c->theta);
        memcpy(m->V[li]+((int64_t)pos*KH+h)*HD,vr+h*HD,(size_t)HD*4);
    }
    float scale=1.f/sqrtf((float)HD); int rep=QH/KH;
    for(int h=0;h<QH;h++){ int kh=h/rep; const float *qh=q+h*HD;
        for(int t=0;t<=pos;t++){ const float *kk=m->K[li]+((int64_t)t*KH+kh)*HD; float a=0; for(int d=0;d<HD;d++) a+=qh[d]*kk[d]; scores[t]=a*scale; }
        softmax(scores,pos+1); float *ch=ctx+h*HD; memset(ch,0,(size_t)HD*4);
        for(int t=0;t<=pos;t++){ const float *vv=m->V[li]+((int64_t)t*KH+kh)*HD; float w=scores[t]; for(int d=0;d<HD;d++) ch[d]+=w*vv[d]; }
        const float *gate=qr+h*2*HD+HD; for(int d=0;d<HD;d++) ch[d]*=sigmoidf1(gate[d]);
    }
    matvec(out,ctx,&l->o); free(qr);free(kr);free(vr);free(q);free(ctx);free(scores);
}

static void delta_attention(Model *m,Layer *l,const float *x,float *out){
    Cfg *c=&m->c; int KH=c->lk_heads,VH=c->lv_heads,KD=c->lk_dim,VD=c->lv_dim;
    int key=KH*KD,val=VH*VD,conv=2*key+val,K=c->conv_kernel,rep=VH/KH;
    float *mix=falloc(conv),*z=falloc(val),*bb=falloc(VH),*aa=falloc(VH),*core=falloc(val);
    matvec(mix,x,&l->qkv); matvec(z,x,&l->z); matvec(bb,x,&l->b); matvec(aa,x,&l->a);
    for(int ch=0;ch<conv;ch++){ float *s=l->conv_state+(int64_t)ch*K; memmove(s,s+1,(size_t)(K-1)*4); s[K-1]=mix[ch];
        float a=0; for(int k=0;k<K;k++) a+=l->conv[(int64_t)ch*K+k]*s[k]; mix[ch]=siluf1(a); }
    float qv[256],kv[256],ov[256],dv[256]; if(KD>256||VD>256){fprintf(stderr,"linear head dimension too large\n");exit(1);}
    for(int h=0;h<VH;h++){ int kh=h/rep; l2_normalize(qv,mix+kh*KD,KD); l2_normalize(kv,mix+key+kh*KD,KD);
        const float *vv=mix+2*key+h*VD; float *state=l->rec_state+(int64_t)h*KD*VD;
        float decay=expf(-expf(l->A_log[h])*softplusf1(aa[h]+l->dt_bias[h])); float beta=sigmoidf1(bb[h]);
        for(int k=0;k<KD;k++) for(int v=0;v<VD;v++) state[(int64_t)k*VD+v]*=decay;
        for(int v=0;v<VD;v++){ float mem=0; for(int k=0;k<KD;k++) mem+=state[(int64_t)k*VD+v]*kv[k]; dv[v]=(vv[v]-mem)*beta; }
        for(int k=0;k<KD;k++) for(int v=0;v<VD;v++) state[(int64_t)k*VD+v]+=kv[k]*dv[v];
        float qs=1.f/sqrtf((float)KD); for(int v=0;v<VD;v++){ float a=0; for(int k=0;k<KD;k++) a+=state[(int64_t)k*VD+v]*(qv[k]*qs); ov[v]=a; }
        rms_gated(core+h*VD,ov,l->lin_norm,z+h*VD,VD,c->eps);
    }
    matvec(out,core,&l->out); free(mix);free(z);free(bb);free(aa);free(core);
}

static ESlot *expert_get(Model *m,int li,int eid){
    ESlot *slots=m->cache[li]; int n=m->cache_n[li];
    for(int i=0;i<n;i++) if(slots[i].eid==eid){m->hits++;slots[i].used=++m->clock;return &slots[i];}
    m->misses++; int at;
    if(n<m->cache_cap){at=n;m->cache_n[li]++;}
    else {at=0;for(int i=1;i<n;i++)if(slots[i].used<slots[at].used)at=i;qt_free(&slots[at].gate);qt_free(&slots[at].up);qt_free(&slots[at].down);}
    ESlot *s=&slots[at]; Cfg *c=&m->c; char name[384];
    snprintf(name,sizeof(name),"model.layers.%d.mlp.experts.%d.gate_proj.weight",li,eid);s->gate=qt_load(m,name,c->moe_inter,c->hidden,0);
    snprintf(name,sizeof(name),"model.layers.%d.mlp.experts.%d.up_proj.weight",li,eid);s->up=qt_load(m,name,c->moe_inter,c->hidden,0);
    snprintf(name,sizeof(name),"model.layers.%d.mlp.experts.%d.down_proj.weight",li,eid);s->down=qt_load(m,name,c->hidden,c->moe_inter,0);
    s->eid=eid;s->used=++m->clock;return s;
}

static void moe(Model *m,Layer *l,int li,const float *x,float *out){
    Cfg *c=&m->c; int E=c->experts,K=g_topk>0&&g_topk<c->topk?g_topk:c->topk,D=c->hidden,I=c->moe_inter,SI=c->shared_inter;
    float *p=falloc(E); matmul_f(p,x,l->router,D,E); softmax(p,E);
    int idx[64]; float w[64]; if(K>64){fprintf(stderr,"top-k too large\n");exit(1);}
    for(int k=0;k<K;k++){int best=-1;float bv=-1;for(int e=0;e<E;e++){int seen=0;for(int j=0;j<k;j++)if(idx[j]==e)seen=1;if(!seen&&p[e]>bv){bv=p[e];best=e;}}idx[k]=best;w[k]=bv;}
    float sum=0;for(int k=0;k<K;k++)sum+=w[k];for(int k=0;k<K;k++)w[k]/=sum;
    float *g=falloc(I>SI?I:SI),*u=falloc(I>SI?I:SI),*h=falloc(D),control;
    matvec(g,x,&l->sh_gate);matvec(u,x,&l->sh_up);for(int i=0;i<SI;i++)g[i]=siluf1(g[i])*u[i];matvec(out,g,&l->sh_down);matvec(&control,x,&l->sh_control);
    float sw=sigmoidf1(control);for(int d=0;d<D;d++)out[d]*=sw;
    for(int k=0;k<K;k++){ESlot *e=expert_get(m,li,idx[k]);matvec(g,x,&e->gate);matvec(u,x,&e->up);for(int i=0;i<I;i++)g[i]=siluf1(g[i])*u[i];matvec(h,g,&e->down);for(int d=0;d<D;d++)out[d]+=w[k]*h[d];}
    free(p);free(g);free(u);free(h);
}

static float *step_token(Model *m,int token,int pos){
    Cfg *c=&m->c; if(pos>=m->max_ctx){fprintf(stderr,"context limit %d reached; set CTX before launch\n",m->max_ctx);exit(1);}
    int D=c->hidden; float *x=falloc(D),*n=falloc(D),*tmp=falloc(D); embed_row(m,token,x);
    for(int li=0;li<c->layers;li++){Layer *l=&m->L[li];rms_centered(n,x,l->in_norm,D,c->eps);
        if(l->full)full_attention(m,l,li,n,pos,tmp);else delta_attention(m,l,n,tmp);for(int d=0;d<D;d++)x[d]+=tmp[d];
        rms_centered(n,x,l->post_norm,D,c->eps);moe(m,l,li,n,tmp);for(int d=0;d<D;d++)x[d]+=tmp[d];}
    rms_centered(n,x,m->final_norm,D,c->eps);float *logits=falloc(c->vocab);matvec(logits,n,&m->lm_head);free(x);free(n);free(tmp);return logits;
}

static double rndu(void){g_rng^=g_rng<<13;g_rng^=g_rng>>7;g_rng^=g_rng<<17;return(double)(g_rng>>11)*(1.0/9007199254740992.0);}
static float *sort_p; static int cmp_prob(const void *a,const void *b){float x=sort_p[*(const int*)a],y=sort_p[*(const int*)b];return x<y?1:x>y?-1:0;}
static int pick_token(const float *logits,int V){
    if(g_temp<=0){int best=0;for(int i=1;i<V;i++)if(logits[i]>logits[best])best=i;return best;}
    float *p=falloc(V);int *idx=xmalloc((size_t)V*sizeof(int));float mx=logits[0];for(int i=1;i<V;i++)if(logits[i]>mx)mx=logits[i];double sum=0;
    for(int i=0;i<V;i++){p[i]=expf((logits[i]-mx)/g_temp);sum+=p[i];idx[i]=i;}for(int i=0;i<V;i++)p[i]/=(float)sum;
    sort_p=p;qsort(idx,(size_t)V,sizeof(int),cmp_prob);double keep=0;int n=V;for(int i=0;i<V;i++){keep+=p[idx[i]];if(keep>=g_topp){n=i+1;break;}}
    double r=rndu()*keep,cum=0;int tok=idx[n-1];for(int i=0;i<n;i++){cum+=p[idx[i]];if(cum>=r){tok=idx[i];break;}}free(p);free(idx);return tok;
}

static int feed_text(Model *m,Tok *tok,const char *text,int *pos,float **logits){
    int cap=(int)strlen(text)+16;int *ids=xmalloc((size_t)cap*sizeof(int));int n=tok_encode(tok,text,(int)strlen(text),ids,cap);
    for(int i=0;i<n;i++){free(*logits);*logits=step_token(m,ids[i],(*pos)++);}free(ids);return n;
}
static int generate_stream(Model *m,Tok *tok,int *pos,float **logits,int limit,int quiet){
    int made=0;for(;made<limit;made++){int id=pick_token(*logits,m->c.vocab);free(*logits);*logits=step_token(m,id,(*pos)++);if(id==m->c.eos)break;
        if(!quiet){char b[512];int n=tok_decode(tok,&id,1,b,sizeof(b)-1);b[n]=0;fwrite(b,1,(size_t)n,stdout);fflush(stdout);}}
    return made;
}

static void run_prompt(Model *m,const char *snap,const char *prompt){
    char path[2048];snprintf(path,sizeof(path),"%s/tokenizer.json",snap);Tok tok;tok_load(&tok,path);int pos=0;float *logits=NULL;
    double pt=now_s();int np=feed_text(m,&tok,prompt,&pos,&logits);double pdt=now_s()-pt;
    fprintf(stderr,"prompt: %d tokens in %.1fs (%.3f tok/s) | context %d | generating up to %d\n",np,pdt,np/(pdt?pdt:1),m->max_ctx,g_ngen);
    double t=now_s();int n=generate_stream(m,&tok,&pos,&logits,g_ngen,0);double dt=now_s()-t;free(logits);
    fprintf(stderr,"\n%d tokens in %.1fs (%.3f tok/s) | expert hit %.1f%% | RSS %.2f GB\n",n,dt,n/(dt?dt:1),m->hits+m->misses?100.0*m->hits/(m->hits+m->misses):0,rss_gb());
}

static void run_serve(Model *m,const char *snap){
    char path[2048];snprintf(path,sizeof(path),"%s/tokenizer.json",snap);Tok tok;tok_load(&tok,path);int pos=0;float *logits=NULL;char *line=NULL;size_t cap=0;int first=1;
    printf("\x01\x01" "READY" "\x01\x01\nSTAT 0 0 0 %.2f\n",rss_gb());fflush(stdout);
    while(getline(&line,&cap,stdin)>=0){size_t n=strlen(line);while(n&&(line[n-1]=='\n'||line[n-1]=='\r'))line[--n]=0;
        if(line[0]==2&&!strcmp(line+1,"RESET")){model_reset(m);pos=0;first=1;free(logits);logits=NULL;printf("\x01\x01" "END" "\x01\x01\nSTAT 0 0 0 %.2f\n",rss_gb());fflush(stdout);continue;}
        int prompt_tokens=0;if(!(line[0]==2&&!strcmp(line+1,"MORE"))){
            size_t need=n+512;char *turn=xmalloc(need);snprintf(turn,need,"%s<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n",first?"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n":"",line);
            prompt_tokens=feed_text(m,&tok,turn,&pos,&logits);free(turn);first=0;}
        if(!logits){printf("\x01\x01" "END" "\x01\x01\nSTAT 0 0 0 %.2f\n",rss_gb());fflush(stdout);continue;}
        uint64_t h0=m->hits,ms0=m->misses;double t=now_s();int made=generate_stream(m,&tok,&pos,&logits,g_ngen,0);double dt=now_s()-t,dh=m->hits-h0,dm=m->misses-ms0;
        printf("\n\x01\x01" "END" "\x01\x01\nSTAT %d %.3f %.1f %.2f\n",made,made/(dt?dt:1),(dh+dm)?100*dh/(dh+dm):0,rss_gb());fflush(stdout);(void)prompt_tokens;
    }free(line);free(logits);
}

/* Sequential implementation of Colibri's multiplex API wire protocol. */
static void run_serve_batch(Model *m,const char *snap){
    char path[2048];snprintf(path,sizeof(path),"%s/tokenizer.json",snap);Tok tok;tok_load(&tok,path);
    char *line=NULL;size_t linecap=0;
    printf("\x01\x01" "READY" "\x01\x01\nSTAT 0 0 0 %.2f\n",rss_gb());fflush(stdout);
    while(getline(&line,&linecap,stdin)>=0){
        unsigned long long rid=0;int slot=0,maximum=0;size_t bytes=0;float temp=0,top_p=1;
        if(!strncmp(line,"CANCEL ",7))continue;
        if(sscanf(line,"SUBMIT %llu %d %zu %d %f %f",&rid,&slot,&bytes,&maximum,&temp,&top_p)!=6){
            fprintf(stdout,"ERROR 0 malformed request\n");fflush(stdout);continue;
        }
        (void)slot;char *prompt=xmalloc(bytes+1);size_t got=fread(prompt,1,bytes,stdin);prompt[got]=0;
        int term=fgetc(stdin);if(got!=bytes||term!='\n'){fprintf(stdout,"ERROR %llu truncated prompt\n",rid);fflush(stdout);free(prompt);continue;}
        model_reset(m);int pos=0;float *logits=NULL;int prompt_tokens=feed_text(m,&tok,prompt,&pos,&logits);free(prompt);
        float old_temp=g_temp,old_top=g_topp;g_temp=temp;g_topp=top_p;
        uint64_t h0=m->hits,ms0=m->misses;double started=now_s();int made=0,stopped=0;
        if(logits)for(;made<maximum;made++){
            int id=pick_token(logits,m->c.vocab);free(logits);logits=step_token(m,id,pos++);
            if(id==m->c.eos){stopped=1;break;}
            char decoded[512];int n=tok_decode(&tok,&id,1,decoded,sizeof(decoded));
            printf("DATA %llu %d\n",rid,n);if(n)fwrite(decoded,1,(size_t)n,stdout);fputc('\n',stdout);fflush(stdout);
        }
        double elapsed=now_s()-started,dh=m->hits-h0,dm=m->misses-ms0;
        printf("DONE %llu STAT %d %.3f %.1f %.2f %d %d\n",rid,made,made/(elapsed?elapsed:1),
               (dh+dm)?100*dh/(dh+dm):0,rss_gb(),prompt_tokens,!stopped&&made>=maximum);fflush(stdout);
        free(logits);g_temp=old_temp;g_topp=old_top;
    }
    free(line);
}

static void run_ids(Model *m,const char *text);

int main(int argc,char **argv){
    const char *snap=getenv("SNAP");if(!snap){fprintf(stderr,"set SNAP to the converted model directory\n");return 2;}
#ifdef _WIN32
    _setmode(_fileno(stdin),_O_BINARY);_setmode(_fileno(stdout),_O_BINARY);
#endif
    int cap=argc>1?atoi(argv[1]):64,maxctx=getenv("CTX")?atoi(getenv("CTX")):4096;if(maxctx<1)maxctx=4096;
    if(getenv("TEMP"))g_temp=(float)atof(getenv("TEMP"));if(getenv("NUCLEUS"))g_topp=(float)atof(getenv("NUCLEUS"));
    if(getenv("TOPK"))g_topk=atoi(getenv("TOPK"));if(getenv("NGEN"))g_ngen=atoi(getenv("NGEN"));
    fprintf(stderr,"Colibri Qwen3.6 CPU | cache %d experts/layer | ctx %d | OpenMP %d threads\n",cap,maxctx,omp_get_max_threads());
    double t=now_s();Model m;model_init(&m,snap,cap,maxctx);fprintf(stderr,"ready in %.1fs | resident dense %.2f GB | RSS %.2f GB\n",now_s()-t,m.resident_bytes/1e9,rss_gb());
    if(getenv("IDS"))run_ids(&m,getenv("IDS"));else if(getenv("SERVE_BATCH"))run_serve_batch(&m,snap);else if(getenv("SERVE"))run_serve(&m,snap);else if(getenv("PROMPT"))run_prompt(&m,snap,getenv("PROMPT"));else fprintf(stderr,"set IDS, PROMPT, or SERVE=1\n");return 0;
}
static void run_ids(Model *m,const char *text){
    char *copy=strdup(text),*p=copy; int pos=0;
    if(!copy){fprintf(stderr,"out of memory\n");exit(1);}
    while(*p){
        char *end=NULL; long id=strtol(p,&end,10);
        if(end==p||id<0||id>=m->c.vocab){fprintf(stderr,"bad IDS list near: %s\n",p);exit(2);}
        float *logits=step_token(m,(int)id,pos++); int best=0;
        for(int i=1;i<m->c.vocab;i++) if(logits[i]>logits[best]) best=i;
        printf("PRED %d\n",best); free(logits); p=end;
        if(*p==',')p++; else if(*p){fprintf(stderr,"IDS must be comma-separated\n");exit(2);}
    }
    free(copy);
}
