CXX = g++

BUILDDIR = build
IDIR = include
OBJDIR = build/o
SRCDIR = src

CXXFLAGS = -I$(IDIR) -std=c++11 -pthread -O3 -Wall -pedantic

OBJS_ = bhtree.o compute_p.o file_reading.o file_writing.o main.o tsne.o util.o vptree.o
OBJS = $(patsubst %,$(OBJDIR)/%,$(OBJS_))

all: tsne

tsne: $(OBJS)
	mkdir -p $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(BUILDDIR)/tsne

$(OBJDIR)/bhtree.o: $(SRCDIR)/bhtree.cpp
	mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $(SRCDIR)/bhtree.cpp -o $@

$(OBJDIR)/compute_p.o: $(SRCDIR)/compute_p.cpp
	mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $(SRCDIR)/compute_p.cpp -o $@

$(OBJDIR)/file_reading.o: $(SRCDIR)/file_reading.cpp
	mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $(SRCDIR)/file_reading.cpp -o $@

$(OBJDIR)/file_writing.o: $(SRCDIR)/file_writing.cpp
	mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $(SRCDIR)/file_writing.cpp -o $@

$(OBJDIR)/main.o: $(SRCDIR)/main.cpp
	mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $(SRCDIR)/main.cpp -o $@

$(OBJDIR)/tsne.o: $(SRCDIR)/tsne.cpp
	mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $(SRCDIR)/tsne.cpp -o $@

$(OBJDIR)/util.o: $(SRCDIR)/util.cpp
	mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $(SRCDIR)/util.cpp -o $@

$(OBJDIR)/vptree.o: $(SRCDIR)/vptree.cpp
	mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $(SRCDIR)/vptree.cpp -o $@

.PHONY: clean

clean:
	\rm $(OBJDIR)/*.o $(BUILDDIR)/tsne -rf $(OBJDIR)