#include <iostream>
#include <algorithm>
#include <cstdint>
#include <random>
#include <easycl/EasyCL.h>

int main(int argc, char *argv[]) 
{
	const int test_size = 128;

	std::random_device rd;
	std::seed_seq s{ rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd() };
	std::mt19937 mt(s);

	EasyCL *cl = EasyCL::createForFirstGpuOtherwiseCpu();

	CLKernel *kern = cl->buildKernel("test.cl", "test");

	std::vector<uint32_t> inbuf(test_size*4), outbuf(test_size), compare(test_size);
	std::generate(inbuf.begin(), inbuf.end(), mt);

	std::cout << "Running CL implementation" << std::endl;
	kern->in(inbuf.size(), &inbuf[0]);
	kern->out(outbuf.size(), &outbuf[0]);
	size_t global_size[] = { test_size };
	kern->run(1, global_size, nullptr);
	delete kern;

	std::cout << "Running local implementation" << std::endl;
	for (int i = 0; i < compare.size(); ++i) {
		compare[i] = inbuf[i] ^ inbuf[i + 1] ^ inbuf[i + 2] ^ inbuf[i + 3];
	}

	std::cout << "Comparing CL test with local implementation" << std::endl;
	for (int i = 0; i < compare.size(); ++i) {
		if (outbuf[i] != compare[i]) {
			std::cout << "Error in index " << i << " " << outbuf[i] << " != " << compare[i] << std::endl;
		}
	}

	return 0;
}