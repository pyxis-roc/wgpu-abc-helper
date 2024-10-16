# SPDX-FileCopyrightText: 2024 University of Rochester
#
# SPDX-License-Identifier: MIT

.PHONY: all clean

CXX = g++
CXXFLAGS = -I../target/release/include
LDFLAGS = -L../target/release -labc_helper -Wl,-rpath,../target/release

BUILD_DIR ?= build

$(BUILD_DIR)/test.out: main.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(BUILD_DIR)/test.out