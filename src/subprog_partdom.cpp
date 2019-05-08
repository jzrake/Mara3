#include <iostream>
#include "ndh5.hpp"
#include "ndarray.hpp"
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
template<std::size_t Rank>
auto make_vertex_coordinate_array(nd::access_pattern_t<Rank> cells_region)
{
    auto vertices_shape = cells_region.shape();

    auto mapping = [] (auto index)
    {
        return mara::spatial_coordinate_t<Rank>();
    };
    return nd::make_array(mapping, vertices_shape);
}




//=============================================================================
class subprog_partdom : public mara::sub_program_t
{
public:

    int main(int argc, const char* argv[]) override
    {
        auto args = mara::argv_to_string_map(argc, argv);
        auto opts = config_template().create().update(args);
        auto blocks_shape = mara::propose_block_decomposition<3>(opts.get<int>("procs"));
        auto domain_shape = nd::make_uniform_shape<3>(opts.get<int>("N"));
        auto blocks = mara::create_access_pattern_array(domain_shape, blocks_shape);

        for (auto index : blocks.indexes())
        {
            std::cout << mara::to_string(index) << " ... " << mara::to_string(blocks(index)) << std::endl;
        }

        auto arrays = blocks | nd::transform([] (auto region)
        {
            return make_vertex_coordinate_array(region);
        });


        auto x = mara::spatial_coordinate_t<3> {1, 2, 3};
        std::cout << mara::to_string(x + x) << std::endl;
        std::cout << mara::to_string(x * 2) << std::endl;


        //auto x = spatial_coordinate_t<3>(blocks_shape);

        // auto space = h5::Dataspace::simple(domain_shape);
        // auto file = h5::File("test.h5", "w");
        // file.require_dataset("data", h5::Datatype::native_double(), space);


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
