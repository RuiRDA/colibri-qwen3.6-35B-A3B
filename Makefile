.PHONY: all qwen36 glm portable test check cuda-test clean install uninstall

all qwen36 glm portable test check cuda-test clean install uninstall:
	$(MAKE) -C c $@