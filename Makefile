# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Default target executed when no arguments are given to make.
default_target: all

.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /workspaces/Image-Contrast-Enhancement

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /workspaces/Image-Contrast-Enhancement

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache

.PHONY : rebuild_cache/fast

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "No interactive CMake dialog available..."
	/usr/bin/cmake -E echo No\ interactive\ CMake\ dialog\ available.
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache

.PHONY : edit_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /workspaces/Image-Contrast-Enhancement/CMakeFiles /workspaces/Image-Contrast-Enhancement/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /workspaces/Image-Contrast-Enhancement/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean

.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named main

# Build rule for target.
main: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 main
.PHONY : main

# fast build rule for target.
main/fast:
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/build
.PHONY : main/fast

src/AGCIE.o: src/AGCIE.cpp.o

.PHONY : src/AGCIE.o

# target to build an object file
src/AGCIE.cpp.o:
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/AGCIE.cpp.o
.PHONY : src/AGCIE.cpp.o

src/AGCIE.i: src/AGCIE.cpp.i

.PHONY : src/AGCIE.i

# target to preprocess a source file
src/AGCIE.cpp.i:
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/AGCIE.cpp.i
.PHONY : src/AGCIE.cpp.i

src/AGCIE.s: src/AGCIE.cpp.s

.PHONY : src/AGCIE.s

# target to generate assembly for a file
src/AGCIE.cpp.s:
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/AGCIE.cpp.s
.PHONY : src/AGCIE.cpp.s

src/AGCWD.o: src/AGCWD.cpp.o

.PHONY : src/AGCWD.o

# target to build an object file
src/AGCWD.cpp.o:
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/AGCWD.cpp.o
.PHONY : src/AGCWD.cpp.o

src/AGCWD.i: src/AGCWD.cpp.i

.PHONY : src/AGCWD.i

# target to preprocess a source file
src/AGCWD.cpp.i:
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/AGCWD.cpp.i
.PHONY : src/AGCWD.cpp.i

src/AGCWD.s: src/AGCWD.cpp.s

.PHONY : src/AGCWD.s

# target to generate assembly for a file
src/AGCWD.cpp.s:
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/AGCWD.cpp.s
.PHONY : src/AGCWD.cpp.s

src/AINDANE.o: src/AINDANE.cpp.o

.PHONY : src/AINDANE.o

# target to build an object file
src/AINDANE.cpp.o:
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/AINDANE.cpp.o
.PHONY : src/AINDANE.cpp.o

src/AINDANE.i: src/AINDANE.cpp.i

.PHONY : src/AINDANE.i

# target to preprocess a source file
src/AINDANE.cpp.i:
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/AINDANE.cpp.i
.PHONY : src/AINDANE.cpp.i

src/AINDANE.s: src/AINDANE.cpp.s

.PHONY : src/AINDANE.s

# target to generate assembly for a file
src/AINDANE.cpp.s:
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/AINDANE.cpp.s
.PHONY : src/AINDANE.cpp.s

src/CEusingLuminanceAdaptation.o: src/CEusingLuminanceAdaptation.cpp.o

.PHONY : src/CEusingLuminanceAdaptation.o

# target to build an object file
src/CEusingLuminanceAdaptation.cpp.o:
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/CEusingLuminanceAdaptation.cpp.o
.PHONY : src/CEusingLuminanceAdaptation.cpp.o

src/CEusingLuminanceAdaptation.i: src/CEusingLuminanceAdaptation.cpp.i

.PHONY : src/CEusingLuminanceAdaptation.i

# target to preprocess a source file
src/CEusingLuminanceAdaptation.cpp.i:
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/CEusingLuminanceAdaptation.cpp.i
.PHONY : src/CEusingLuminanceAdaptation.cpp.i

src/CEusingLuminanceAdaptation.s: src/CEusingLuminanceAdaptation.cpp.s

.PHONY : src/CEusingLuminanceAdaptation.s

# target to generate assembly for a file
src/CEusingLuminanceAdaptation.cpp.s:
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/CEusingLuminanceAdaptation.cpp.s
.PHONY : src/CEusingLuminanceAdaptation.cpp.s

src/GCEHistMod.o: src/GCEHistMod.cpp.o

.PHONY : src/GCEHistMod.o

# target to build an object file
src/GCEHistMod.cpp.o:
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/GCEHistMod.cpp.o
.PHONY : src/GCEHistMod.cpp.o

src/GCEHistMod.i: src/GCEHistMod.cpp.i

.PHONY : src/GCEHistMod.i

# target to preprocess a source file
src/GCEHistMod.cpp.i:
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/GCEHistMod.cpp.i
.PHONY : src/GCEHistMod.cpp.i

src/GCEHistMod.s: src/GCEHistMod.cpp.s

.PHONY : src/GCEHistMod.s

# target to generate assembly for a file
src/GCEHistMod.cpp.s:
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/GCEHistMod.cpp.s
.PHONY : src/GCEHistMod.cpp.s

src/IAGCWD.o: src/IAGCWD.cpp.o

.PHONY : src/IAGCWD.o

# target to build an object file
src/IAGCWD.cpp.o:
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/IAGCWD.cpp.o
.PHONY : src/IAGCWD.cpp.o

src/IAGCWD.i: src/IAGCWD.cpp.i

.PHONY : src/IAGCWD.i

# target to preprocess a source file
src/IAGCWD.cpp.i:
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/IAGCWD.cpp.i
.PHONY : src/IAGCWD.cpp.i

src/IAGCWD.s: src/IAGCWD.cpp.s

.PHONY : src/IAGCWD.s

# target to generate assembly for a file
src/IAGCWD.cpp.s:
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/IAGCWD.cpp.s
.PHONY : src/IAGCWD.cpp.s

src/JHE.o: src/JHE.cpp.o

.PHONY : src/JHE.o

# target to build an object file
src/JHE.cpp.o:
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/JHE.cpp.o
.PHONY : src/JHE.cpp.o

src/JHE.i: src/JHE.cpp.i

.PHONY : src/JHE.i

# target to preprocess a source file
src/JHE.cpp.i:
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/JHE.cpp.i
.PHONY : src/JHE.cpp.i

src/JHE.s: src/JHE.cpp.s

.PHONY : src/JHE.s

# target to generate assembly for a file
src/JHE.cpp.s:
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/JHE.cpp.s
.PHONY : src/JHE.cpp.s

src/LDR.o: src/LDR.cpp.o

.PHONY : src/LDR.o

# target to build an object file
src/LDR.cpp.o:
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/LDR.cpp.o
.PHONY : src/LDR.cpp.o

src/LDR.i: src/LDR.cpp.i

.PHONY : src/LDR.i

# target to preprocess a source file
src/LDR.cpp.i:
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/LDR.cpp.i
.PHONY : src/LDR.cpp.i

src/LDR.s: src/LDR.cpp.s

.PHONY : src/LDR.s

# target to generate assembly for a file
src/LDR.cpp.s:
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/LDR.cpp.s
.PHONY : src/LDR.cpp.s

src/SEF.o: src/SEF.cpp.o

.PHONY : src/SEF.o

# target to build an object file
src/SEF.cpp.o:
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/SEF.cpp.o
.PHONY : src/SEF.cpp.o

src/SEF.i: src/SEF.cpp.i

.PHONY : src/SEF.i

# target to preprocess a source file
src/SEF.cpp.i:
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/SEF.cpp.i
.PHONY : src/SEF.cpp.i

src/SEF.s: src/SEF.cpp.s

.PHONY : src/SEF.s

# target to generate assembly for a file
src/SEF.cpp.s:
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/SEF.cpp.s
.PHONY : src/SEF.cpp.s

src/WTHE.o: src/WTHE.cpp.o

.PHONY : src/WTHE.o

# target to build an object file
src/WTHE.cpp.o:
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/WTHE.cpp.o
.PHONY : src/WTHE.cpp.o

src/WTHE.i: src/WTHE.cpp.i

.PHONY : src/WTHE.i

# target to preprocess a source file
src/WTHE.cpp.i:
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/WTHE.cpp.i
.PHONY : src/WTHE.cpp.i

src/WTHE.s: src/WTHE.cpp.s

.PHONY : src/WTHE.s

# target to generate assembly for a file
src/WTHE.cpp.s:
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/WTHE.cpp.s
.PHONY : src/WTHE.cpp.s

src/Ying_2017_CAIP.o: src/Ying_2017_CAIP.cpp.o

.PHONY : src/Ying_2017_CAIP.o

# target to build an object file
src/Ying_2017_CAIP.cpp.o:
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/Ying_2017_CAIP.cpp.o
.PHONY : src/Ying_2017_CAIP.cpp.o

src/Ying_2017_CAIP.i: src/Ying_2017_CAIP.cpp.i

.PHONY : src/Ying_2017_CAIP.i

# target to preprocess a source file
src/Ying_2017_CAIP.cpp.i:
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/Ying_2017_CAIP.cpp.i
.PHONY : src/Ying_2017_CAIP.cpp.i

src/Ying_2017_CAIP.s: src/Ying_2017_CAIP.cpp.s

.PHONY : src/Ying_2017_CAIP.s

# target to generate assembly for a file
src/Ying_2017_CAIP.cpp.s:
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/Ying_2017_CAIP.cpp.s
.PHONY : src/Ying_2017_CAIP.cpp.s

src/adaptiveImageEnhancement.o: src/adaptiveImageEnhancement.cpp.o

.PHONY : src/adaptiveImageEnhancement.o

# target to build an object file
src/adaptiveImageEnhancement.cpp.o:
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/adaptiveImageEnhancement.cpp.o
.PHONY : src/adaptiveImageEnhancement.cpp.o

src/adaptiveImageEnhancement.i: src/adaptiveImageEnhancement.cpp.i

.PHONY : src/adaptiveImageEnhancement.i

# target to preprocess a source file
src/adaptiveImageEnhancement.cpp.i:
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/adaptiveImageEnhancement.cpp.i
.PHONY : src/adaptiveImageEnhancement.cpp.i

src/adaptiveImageEnhancement.s: src/adaptiveImageEnhancement.cpp.s

.PHONY : src/adaptiveImageEnhancement.s

# target to generate assembly for a file
src/adaptiveImageEnhancement.cpp.s:
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/adaptiveImageEnhancement.cpp.s
.PHONY : src/adaptiveImageEnhancement.cpp.s

src/main.o: src/main.cpp.o

.PHONY : src/main.o

# target to build an object file
src/main.cpp.o:
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/main.cpp.o
.PHONY : src/main.cpp.o

src/main.i: src/main.cpp.i

.PHONY : src/main.i

# target to preprocess a source file
src/main.cpp.i:
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/main.cpp.i
.PHONY : src/main.cpp.i

src/main.s: src/main.cpp.s

.PHONY : src/main.s

# target to generate assembly for a file
src/main.cpp.s:
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/main.cpp.s
.PHONY : src/main.cpp.s

src/util.o: src/util.cpp.o

.PHONY : src/util.o

# target to build an object file
src/util.cpp.o:
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/util.cpp.o
.PHONY : src/util.cpp.o

src/util.i: src/util.cpp.i

.PHONY : src/util.i

# target to preprocess a source file
src/util.cpp.i:
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/util.cpp.i
.PHONY : src/util.cpp.i

src/util.s: src/util.cpp.s

.PHONY : src/util.s

# target to generate assembly for a file
src/util.cpp.s:
	$(MAKE) -f CMakeFiles/main.dir/build.make CMakeFiles/main.dir/src/util.cpp.s
.PHONY : src/util.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... rebuild_cache"
	@echo "... edit_cache"
	@echo "... main"
	@echo "... src/AGCIE.o"
	@echo "... src/AGCIE.i"
	@echo "... src/AGCIE.s"
	@echo "... src/AGCWD.o"
	@echo "... src/AGCWD.i"
	@echo "... src/AGCWD.s"
	@echo "... src/AINDANE.o"
	@echo "... src/AINDANE.i"
	@echo "... src/AINDANE.s"
	@echo "... src/CEusingLuminanceAdaptation.o"
	@echo "... src/CEusingLuminanceAdaptation.i"
	@echo "... src/CEusingLuminanceAdaptation.s"
	@echo "... src/GCEHistMod.o"
	@echo "... src/GCEHistMod.i"
	@echo "... src/GCEHistMod.s"
	@echo "... src/IAGCWD.o"
	@echo "... src/IAGCWD.i"
	@echo "... src/IAGCWD.s"
	@echo "... src/JHE.o"
	@echo "... src/JHE.i"
	@echo "... src/JHE.s"
	@echo "... src/LDR.o"
	@echo "... src/LDR.i"
	@echo "... src/LDR.s"
	@echo "... src/SEF.o"
	@echo "... src/SEF.i"
	@echo "... src/SEF.s"
	@echo "... src/WTHE.o"
	@echo "... src/WTHE.i"
	@echo "... src/WTHE.s"
	@echo "... src/Ying_2017_CAIP.o"
	@echo "... src/Ying_2017_CAIP.i"
	@echo "... src/Ying_2017_CAIP.s"
	@echo "... src/adaptiveImageEnhancement.o"
	@echo "... src/adaptiveImageEnhancement.i"
	@echo "... src/adaptiveImageEnhancement.s"
	@echo "... src/main.o"
	@echo "... src/main.i"
	@echo "... src/main.s"
	@echo "... src/util.o"
	@echo "... src/util.i"
	@echo "... src/util.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

