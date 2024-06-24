# Compiler
CXX = mpiCC

# Compiler flags
CXXFLAGS = -Wall -Wextra -std=c++20 -DOMPI_SKIP_MPICXX

# Executable name
TARGET = main

# Source files
SRCS = main.cpp 2d.cpp 3d.cpp matrix.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Default rule
all: $(TARGET)

# Rule to build the executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

# Rule to build object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up build files
clean:
	rm -f $(TARGET) $(OBJS)

# Phony targets
.PHONY: all clean
