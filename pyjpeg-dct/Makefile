CC=gcc
CFLAGS=-fPIC -Wall -Wextra -pthread -Wno-unused-parameter -std=c99 -g
INCLUDE=
LIBS=-ljpeg -lpython2.7 -lpython3.2
BIN_NAME=test
SRC_PATH = .
BUILD_PATH = .
BIN_PATH = .
LDFLAGS = $(LIBS)

SHELL = /bin/bash

SOURCES = $(shell find $(SRC_PATH)/ -name '*.c')
OBJECTS = $(SOURCES:$(SRC_PATH)/%.c=$(BUILD_PATH)/%.o)

.PHONY: all
all: $(BIN_PATH)/$(BIN_NAME)

.PHONY: clean
clean:
	rm -rf $(BIN_NAME) $(OBJECTS)

$(BUILD_PATH)/%.o: $(SRC_PATH)/%.c $(SRC_PATH)/%.c=%.h
	$(CC) $(CFLAGS) $(LIBS) $(INCLUDE) -c -o $@ $<

$(BIN_PATH)/$(BIN_NAME): $(OBJECTS)
	$(CC) $(OBJECTS) $(LDFLAGS) -g -o $@
