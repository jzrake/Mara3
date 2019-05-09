#include <iostream>
#include "ndarray.hpp"
#include "ndarray_ops.hpp"
#include "ndh5.hpp"
#include "config.hpp"
#include "serialize.hpp"
#include "subprogram.hpp"
#include "parallel.hpp"
#include "datatypes.hpp"




//=============================================================================
static auto config_template()
{
    return mara::make_config_template()
    .item("N", 256)
    .item("procs", 256);
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
            std::cout << mara::to_string(index) << " ... " << mara::to_string(array_of_access_patterns(index)) << std::endl;
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
            | nd::map([   ] (auto j) { return mara::spatial_coordinate_t<3>::from_range(j); });
        });

        for (auto block_index : vertex_arrays.indexes())
        {
            std::cout << "\n<============ vertex array for block index " << mara::to_string(block_index) << " ============>\n";

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
            std::cout << "\n<============ cell-center array for block index " << mara::to_string(block_index) << " ============>\n";

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
