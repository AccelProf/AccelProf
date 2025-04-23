DEBUG ?= 0

all:
	./bin/build DEBUG=$(DEBUG)

clean:
	./bin/clean
