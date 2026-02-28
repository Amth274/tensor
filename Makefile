# ============================================================
# Compiler Settings
# ============================================================

CC      := gcc

# Debug build flags
CFLAGS  := -Wall -Wextra -std=c11 -g -O0 -fsanitize=address
CFLAGS  += -Iinclude -Isrc

# Linker flags (sanitizer must also be here)
# LDFLAGS := -fsanitize=address
LDFLAGS := -fsanitize=address,undefined

TARGET      := tensor
SRC_DIR     := src
BUILD_DIR   := build


# ============================================================
# Source Files
# ============================================================

# All source files inside src/
SRCS := $(shell find $(SRC_DIR) -name '*.c')

# Add test file
SRCS += test.c

# Convert src/foo.c → build/foo.o
OBJS := $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(filter $(SRC_DIR)/%.c,$(SRCS)))
OBJS += $(BUILD_DIR)/test.o


# ============================================================
# Build Rules
# ============================================================

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $@ $(LDFLAGS)


# Compile rule for src files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@


# Compile rule for test.c
$(BUILD_DIR)/test.o: test.c
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@


# ============================================================
# Utility
# ============================================================

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

rebuild: clean all

.PHONY: all clean run rebuild