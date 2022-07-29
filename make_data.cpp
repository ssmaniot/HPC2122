#include <iostream>
#include <fstream>

int main(int argc, char *argv[])
{
    double data[4] = {9.5, -3.4, 1.0, 2.1};
    int i;

    std::ofstream out("test.bin", std::ios::out | std::ios::binary);
    if (!out)
    {
        std::cerr << "Cannot open file.\n";
        return 1;
    }

    out.write((char *)&data, sizeof(data));
    out.close();

    for (i = 0; i < 4; ++i)
        data[i] = 0;

    std::ifstream in("test.bin", std::ios::in | std::ios::binary);
    in.read((char *)&data, sizeof(data));

    // see how many bytes have been read
    std::cout << in.gcount() << " bytes read\n";

    for (i = 0; i < 4; ++i)
        std::cout << data[i] << ' ';
    std::cout << '\n';

    in.close();
    return 0;
}