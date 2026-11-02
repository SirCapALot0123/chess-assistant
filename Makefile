ARCHS = arm64
TARGET = iphone:clang:latest:15.0

THEOS_PACKAGE_SCHEME = rootless

include $(THEOS)/makefiles/common.mk

TWEAK_NAME = Chess

Chess_FILES      = Tweak.xm engine.mm maia.mm
Chess_FRAMEWORKS = UIKit Foundation QuartzCore CoreML
Chess_LIBRARIES  = substrate
Chess_CFLAGS     = -fobjc-arc -Wno-deprecated-declarations -Isf/src -std=c++17
Chess_LDFLAGS    = -Lsf/src -lstockfish -lc++

# Embed the validated Maia 79M Core ML package in the tweak dylib.
MAIA_PACKAGE = tools/maia79m/maia3-79m.mlpackage
ifeq ($(wildcard $(MAIA_PACKAGE)/Manifest.json),)
$(error Maia 79M package missing: run tools/maia79m/convert_validate.py --mode all)
endif
ifeq ($(wildcard $(MAIA_PACKAGE)/Data/com.apple.CoreML/model.mlmodel),)
$(error Maia 79M model file missing: $(MAIA_PACKAGE))
endif
ifeq ($(wildcard $(MAIA_PACKAGE)/Data/com.apple.CoreML/weights/weight.bin),)
$(error Maia 79M weight file missing: $(MAIA_PACKAGE))
endif
Chess_LDFLAGS    += -Wl,-sectcreate,__TEXT,__maia_mfst,$(MAIA_PACKAGE)/Manifest.json
Chess_LDFLAGS    += -Wl,-sectcreate,__TEXT,__maia_model,$(MAIA_PACKAGE)/Data/com.apple.CoreML/model.mlmodel
Chess_LDFLAGS    += -Wl,-sectcreate,__TEXT,__maia_wghts,$(MAIA_PACKAGE)/Data/com.apple.CoreML/weights/weight.bin

include $(THEOS)/makefiles/tweak.mk
