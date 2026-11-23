CXX       = g++
CXXFLAGS  = -g -Wall -Weffc++ -std=c++11 -Iinclude -MMD -MP

SRC_DIR   = src
BIN_DIR   = bin
TARGET    = $(BIN_DIR)/server

SRCS      = $(wildcard $(SRC_DIR)/*.cpp)
OBJS      = $(patsubst $(SRC_DIR)/%.cpp,$(BIN_DIR)/%.o,$(SRCS))
DEPS      = $(OBJS:.o=.d)

# All object files except Main.o (for building standalone unit tests)
TEST_OBJS = $(filter-out $(BIN_DIR)/Main.o,$(OBJS))

.PHONY: all clean tests

all: $(TARGET)

$(TARGET): $(OBJS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

$(BIN_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

tests: $(TEST_OBJS) tests.cpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $(BIN_DIR)/tests tests.cpp $(TEST_OBJS)

clean:
	rm -rf $(BIN_DIR)

-include $(DEPS)