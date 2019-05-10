/**
 ==============================================================================
 Copyright 2019, Jonathan Zrake

 Permission is hereby granted, free of charge, to any person obtaining a copy of
 this software and associated documentation files (the "Software"), to deal in
 the Software without restriction, including without limitation the rights to
 use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 of the Software, and to permit persons to whom the Software is furnished to do
 so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
 
 ==============================================================================
*/




#include <map>
#include <iostream>
#include "app_subprogram.hpp"
#include "core_dimensional.hpp"




//=============================================================================
std::unique_ptr<mara::sub_program_t> make_subprog_boilerlate();
std::unique_ptr<mara::sub_program_t> make_subprog_partdom();
std::unique_ptr<mara::sub_program_t> make_subprog_shockwave();




//=============================================================================
int main(int argc, const char* argv[])
{
    auto programs = std::map<std::string, std::unique_ptr<mara::sub_program_t>>();

    programs["boilerplate"] = make_subprog_boilerlate();
    programs["partdom"]     = make_subprog_partdom();
    // programs["shockwave"]   = make_subprog_shockwave();

    if (argc == 1)
    {
       std::cout << "usages: \n";

       for (auto& prog : programs)
       {
           std::cout << "    mara " << prog.first << std::endl;
       }
       return 0;
    }
    else if (programs.count(argv[1]))
    {
       return programs.at(argv[1])->main(argc - 1, argv + 1);
    }

    std::cout << "invalid sub-program '" << argv[1] << "'\n";
    return 0;
}
