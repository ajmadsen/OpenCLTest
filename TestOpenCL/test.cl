kernel void test(global const uint *in, global uint *out)
{
	const uint x = get_global_id(0);

	out[x] = in[x] ^ in[x+1] ^ in[x+2] ^ in[x+3];
}