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
#include "app_compile_opts.hpp"
#if MARA_COMPILE_SUBPROGRAM_PARTDOM




#include <iostream>
#include "ndarray.hpp"
#include "ndarray_ops.hpp"
#include "ndh5.hpp"
#include "app_config.hpp"
#include "app_serialize.hpp"
#include "app_subprogram.hpp"
#include "app_parallel.hpp"
#include "core_geometric.hpp"




//=============================================================================
static auto config_template()
{
    return mara::make_config_template()
    .item("N", 4)
    .item("procs", 2);
}




//=============================================================================
class subprog_partdom : public mara::sub_program_t
{
public:

    int main(int argc, const char* argv[]) override
    {
        auto opts = config_template()
        .create()
        .update(mara::argv_to_string_map(argc, argv));


        // Demonstrates use of the block decomposition algorithms
        // =====================================================================
        auto blocks_shape = mara::propose_block_decomposition<3>(opts.get<int>("procs"));
        auto domain_shape = nd::make_uniform_shape<3>(opts.get<int>("N"));
        auto array_of_access_patterns = mara::create_access_pattern_array(domain_shape, blocks_shape);

        for (auto index : array_of_access_patterns.indexes())
        {
            std::cout << nd::to_string(index) << " ... " << nd::to_string(array_of_access_patterns(index)) << std::endl;
        }


        // Demonstrates computing the vertex coordinates from the array of
        // access patterns.
        // =====================================================================
        auto vertex_arrays = array_of_access_patterns
        | nd::map([] (auto reg) { return reg.with_final(reg.final.transform([] (auto x) { return x + 1; })); })
        | nd::map([] (auto reg)
        {
            return nd::index_array(reg.shape())
            | nd::map([reg] (auto i) { return reg.map_index(i); })
            | nd::map([   ] (auto j) { return mara::make_spatial_coordinate(j[0], j[1], j[2]); });
        });

        for (auto block_index : vertex_arrays.indexes())
        {
            std::cout << "\n<============ vertex array for block index " << nd::to_string(block_index) << " ============>\n";

            for (auto element_index : vertex_arrays(block_index).indexes())
            {
                std::cout << mara::to_string(vertex_arrays(block_index)(element_index)) << std::endl;
            }
        }


        // Demonstrates getting the cell-center coordinates of each of the
        // vertex arrays
        // =====================================================================
        auto cell_center_arrays = vertex_arrays
        | nd::map([] (auto array)
        {
            return array
            | nd::midpoint_on_axis(0)
            | nd::midpoint_on_axis(1)
            | nd::midpoint_on_axis(2);
        });

        for (auto block_index : cell_center_arrays.indexes())
        {
            std::cout << "\n<============ cell-center array for block index " << nd::to_string(block_index) << " ============>\n";

            for (auto element_index : cell_center_arrays(block_index).indexes())
            {
                std::cout << mara::to_string(cell_center_arrays(block_index)(element_index)) << std::endl;
            }
        }


        // Demonstrates that we can write one of the vertex arrays into a subset
        // of the file
        //=====================================================================
        auto space = h5::Dataspace::simple(domain_shape);
        auto file = h5::File("test.h5", "w");
        auto data = file.require_dataset("data", h5::Datatype::native_double(), space);

        data.write(cell_center_arrays(0, 0, 0)
            | nd::map([] (auto x) { return x[0]; })
            | nd::to_shared(),
            space.select(mara::make_hdf5_hyperslab(array_of_access_patterns(0, 0, 0))));


        return 0;
    }

    std::string name() const override
    {
        return "partdom";
    }
};

std::unique_ptr<mara::sub_program_t> make_subprog_partdom()
{
    return std::make_unique<subprog_partdom>();
}

#endif // MARA_COMPILE_SUBPROGRAM_PARTDOM