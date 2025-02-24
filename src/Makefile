# Useful Resource: https://www.cs.colby.edu/maxwell/courses/tutorials/maketutor/

# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -O3 -Wall -Wextra -std=c++20

# Libraries
LDFLAGS = 

# Source files
SRC = main.cpp 

# Put all the obj files in an `obj` directory so the codebase isn't filled with them
ODIR = obj

# Object files (convert .cpp -> .o and place in ODIR)
OBJ = $(patsubst %.cpp, $(ODIR)/%.o, $(SRC))

# Executable name
TARGET = program

# Default target
all: $(TARGET)

# Link all object files into the final executable
$(TARGET): $(OBJ)
	$(CXX) $(OBJ) -o $(TARGET) $(LDFLAGS)

# Ensure the object directory exists before compiling
$(ODIR)/%.o: %.cpp
	@mkdir -p $(dir $@) 
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up build files
clean:
	rm -rf $(ODIR) $(TARGET)