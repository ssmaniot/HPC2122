#!/usr/bin/bash

g++ -std=c++14 -pedantic -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization           \
       	-Wformat=2 -Winit-self -Wlogical-op -Wmissing-declarations -Wmissing-include-dirs -Wnoexcept -Wold-style-cast \
	-Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-conversion -Wsign-promo -Wstrict-null-sentinel         \
	-Wstrict-overflow=5 -Wswitch-default -Wundef -Wno-unused                                              \
	-O3 -march=native                                                                                             \
	-o slink Slink/slink.cpp src/CSV.cpp -I include/ -fopenmp
