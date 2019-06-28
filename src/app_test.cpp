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
#if MARA_COMPILE_SUBPROGRAM_TEST




#include <array>
#include "core_catch.hpp"
#include "core_hdf5.hpp"
#include "core_tree.hpp"
#include "app_serialize.hpp"
#include "app_serialize_tree.hpp"
#include "app_filesystem.hpp"




//=============================================================================
struct hdf5_compound_test_t
{
    int a = 1;
    double b = 3.14;

    bool operator==(const hdf5_compound_test_t& other) const { return a == other.a && b == other.b; }
};




//=============================================================================
template<>
struct h5::hdf5_type_info<hdf5_compound_test_t>
{
    using native_type = hdf5_compound_test_t;
    static auto make_datatype_for(const native_type& value)
    {
        return h5::Datatype::compound<hdf5_compound_test_t>({
            {"a", offsetof(hdf5_compound_test_t, a), Datatype::native_int()},
            {"b", offsetof(hdf5_compound_test_t, b), Datatype::native_double()},
        });
    }
    static auto make_dataspace_for(const native_type& value) { return Dataspace::scalar(); }
    static auto convert_to_writable(const native_type& value) { return value; }
    static auto prepare(const Datatype&, const Dataspace& space) { return native_type(); }
    static auto finalize(native_type&& value) { return std::move(value); }
    static auto get_address(const native_type& value) { return &value; }
    static auto get_address(native_type& value) { return &value; }
};




//=============================================================================
SCENARIO("Files can be created", "[h5::File]")
{
    GIVEN("A file opened for writing")
    {
        auto file = h5::File("test.h5", "w");

        THEN("The file reports being opened with read/write intent")
        {
            REQUIRE(file.is_open());
            REQUIRE(file.intent() == h5::Intent::rdwr);
        }

        WHEN("The file is closed manually")
        {
            file.close();

            THEN("It reports as not open, and can be closed again without effect")
            {
                REQUIRE_FALSE(file.is_open());
                REQUIRE_NOTHROW(file.close());
            }
        }
        mara::filesystem::remove_file("test.h5");
    }

    GIVEN("A file is opened for reading")
    {
        h5::File("test.h5", "w");
        {
            auto file = h5::File("test.h5", "r");
        
            THEN("It reports as open with read-only intent")
            {
                REQUIRE(file.is_open());
                REQUIRE(file.intent() == h5::Intent::rdonly);
            }
        }
        mara::filesystem::remove_file("test.h5");
    }

    GIVEN("A filename that does not exist")
    {
        THEN("h5::File::exists reports it does not exist, and open as read throws")
        {
            REQUIRE_FALSE(h5::File::exists("no-exist.h5"));
            REQUIRE_THROWS(h5::File("no-exist.h5", "r"));
        }
    }
}


SCENARIO("Groups can be created in files", "[h5::Group]")
{
    GIVEN("A file opened for writing")
    {
        auto file = h5::File("test.h5", "w");

        WHEN("Three groups are created")
        {
            auto group1 = file.require_group("group1");
            auto group2 = file.require_group("group2");
            auto group3 = file.require_group("group3");

            THEN("file.size() returns 3")
            {
                REQUIRE(file.size() == 3);
            }

            THEN("The groups can be opened without throwing, but a non-existent group does throw")
            {
                REQUIRE_NOTHROW(file.open_group("group1"));
                REQUIRE_NOTHROW(file.open_group("group2"));
                REQUIRE_NOTHROW(file.open_group("group3"));
                REQUIRE_THROWS(file.open_group("no-exist"));

                REQUIRE_NOTHROW(file["group1"]);
                REQUIRE_NOTHROW(file["group1"]["new-group"]); // creates a new group
            }

            THEN("The groups have the correct names")
            {
                int n = 0;

                for (auto group : file)
                {
                    switch (n++)
                    {
                        case 0: REQUIRE(group == "group1"); break;
                        case 1: REQUIRE(group == "group2"); break;
                        case 2: REQUIRE(group == "group3"); break;
                    }
                }
            }
        }

        WHEN("The file is closed")
        {
            file.close();

            THEN("Trying to open a group fails")
            {
                REQUIRE_THROWS(file.open_group("group1"));
            }
        }
        mara::filesystem::remove_file("test.h5");
    }
}


SCENARIO("Data types can be created", "[h5::Datatype]")
{
    REQUIRE(h5::make_datatype_for(char()).size() == sizeof(char));
    REQUIRE(h5::make_datatype_for(int()).size() == sizeof(int));
    REQUIRE(h5::make_datatype_for(double()).size() == sizeof(double));
    REQUIRE(h5::make_datatype_for(std::string("message")).size() == 7);
    REQUIRE(h5::make_datatype_for(std::vector<int>()).size() == sizeof(int));
}


SCENARIO("Data spaces can be created", "[h5::Dataspace]")
{
    REQUIRE(h5::Dataspace().size() == 0);
    REQUIRE(h5::Dataspace::scalar().rank() == 0);
    REQUIRE(h5::Dataspace::scalar().size() == 1);
    REQUIRE(h5::Dataspace::scalar().select_all().size() == 1);
    REQUIRE(h5::Dataspace::scalar().select_none().size() == 1);
    REQUIRE(h5::Dataspace::scalar().select_all().selection_size() == 1);
    REQUIRE(h5::Dataspace::scalar().select_none().selection_size() == 0);
    REQUIRE(h5::Dataspace::simple(std::array<int, 3>{10, 10, 10}).rank() == 3);
    REQUIRE(h5::Dataspace::simple(std::array<int, 3>{10, 10, 10}).size() == 1000);
    REQUIRE(h5::Dataspace{10, 21}.size() == 210);
    REQUIRE(h5::Dataspace{10, 21}.selection_size() == 210);
    REQUIRE(h5::Dataspace{10, 21}.selection_lower() == std::vector<std::size_t>{0, 0});
    REQUIRE(h5::Dataspace{10, 21}.selection_upper() == std::vector<std::size_t>{9, 20});
    REQUIRE_NOTHROW(h5::Dataspace::scalar().select_all());
}


SCENARIO("Data sets can be created, read, and written to", "[h5::Dataset]")
{
    GIVEN("A file opened for writing, native double data type, and a scalar data space")
    {
        auto file = h5::File("test.h5", "w");
        auto type = h5::make_datatype_for(double());
        auto space = h5::Dataspace::scalar();
        auto dset = file.require_dataset("data", type, space);

        THEN("The dataset exists in the file with expected properties")
        {
            REQUIRE(file.open_dataset("data").get_type() == type);
            REQUIRE(file.open_dataset("data").get_space().size() == space.size());
            REQUIRE_NOTHROW(file.open_dataset("data"));
            REQUIRE_NOTHROW(file.require_dataset("data", type, space));
            REQUIRE_THROWS(file.require_dataset("data", h5::make_datatype_for(int()), space));
        }
        mara::filesystem::remove_file("test.h5");
    }

    GIVEN("A file opened for writing, native int data type, and a simple data space")
    {
        auto file = h5::File("test.h5", "w");
        auto type = h5::make_datatype_for(int());
        auto space = h5::Dataspace::simple(std::array<int, 1>{4});
        auto dset = file.require_dataset("data", type, space);

        THEN("The dataset exists in the file with expected properties")
        {
            REQUIRE(file.open_dataset("data").get_type() == type);
            REQUIRE(file.open_dataset("data").get_space() == space);
            REQUIRE_NOTHROW(file.open_dataset("data"));
            REQUIRE_NOTHROW(file.require_dataset("data", type, space));
            REQUIRE_THROWS(file.require_dataset("data", h5::make_datatype_for(double()), space));
        }

        WHEN("We have a std::vector<int>{1, 2, 3, 4}")
        {
            auto data = std::vector<int>{1, 2, 3, 4};

            THEN("It can be written to the data set and read back")
            {
                REQUIRE_NOTHROW(dset.write(data));
                REQUIRE(dset.read<std::vector<int>>() == data);
                REQUIRE_THROWS(dset.read<std::vector<double>>());
                REQUIRE_THROWS(dset.read<double>());
            }
        }

        WHEN("We have a std::vector<int>{1, 2, 3} or std::vector<double>{1, 2, 3, 4}")
        {
            auto data1 = std::vector<int>{1, 2, 3};
            auto data2 = std::vector<double>{1, 2, 3, 4};

            THEN("It cannot be written to the data set")
            {
                REQUIRE_THROWS(dset.write(data1));
                REQUIRE_THROWS(dset.write(data2));
            }
        }
        mara::filesystem::remove_file("test.h5");
    }

    GIVEN("A file opened for writing and a double")
    {
        auto data = 10.0;
        auto file = h5::File("test.h5", "w");
        auto type = h5::make_datatype_for(data);
        auto space = h5::Dataspace::scalar();
        auto dset = file.require_dataset("data", type, space);

        THEN("The double can be written to a scalar dataset")
        {
            REQUIRE_NOTHROW(file.require_dataset("data", type, space));
            REQUIRE_NOTHROW(dset.write(data));
            REQUIRE(dset.read<double>() == 10.0);
            REQUIRE_THROWS(dset.read<int>() == 10);
        }
        mara::filesystem::remove_file("test.h5");
    }

    GIVEN("A file opened for writing")
    {
        auto data = std::string("The string value");
        auto file = h5::File("test.h5", "w");
        auto type = h5::make_datatype_for(data);
        auto space = h5::Dataspace::scalar();
        auto dset = file.require_dataset("data1", type, space);

        THEN("A string can be written to a scalar dataset")
        {
            REQUIRE_NOTHROW(file.require_dataset("data1", type, space));
            REQUIRE_NOTHROW(dset.write(data));
            REQUIRE_THROWS(dset.read<int>());
            REQUIRE(dset.read<std::string>() == "The string value");
        }

        WHEN("A string, int, and double are written directly to the file")
        {
            file.write("data2", data);
            file.write("data3", 10.0);
            file.write("data4", 11);

            THEN("They can be read back out again")
            {
                REQUIRE(file.read<std::string>("data2") == data);
                REQUIRE(file.read<double>("data3") == 10.0);
                REQUIRE(file.read<int>("data4") == 11);
            }
        }
        mara::filesystem::remove_file("test.h5");
    }

    GIVEN("A file opened for writing")
    {
        auto file = h5::File("test.h5", "w");

        WHEN("An int and double vector are written to it")
        {
            auto data1 = std::vector<int>{1, 2, 3, 4};
            auto data2 = std::vector<double>{1, 2, 3};

            file.write("data1", data1);
            file.write("data2", data2);

            THEN("They can be read back again")
            {
                REQUIRE(file.read<decltype(data1)>("data1") == data1);
                REQUIRE(file.read<decltype(data2)>("data2") == data2);
            }

            THEN("Trying to read the wrong type throws")
            {
                REQUIRE_THROWS(file.read<decltype(data2)>("data1")); // mismatched data types
                REQUIRE_THROWS(file.read<decltype(data1)>("data2"));
            }
        }
        mara::filesystem::remove_file("test.h5");
    }

    GIVEN("A file opened for writing and a compound data type")
    {
        auto data = hdf5_compound_test_t{};
        auto file = h5::File("test.h5", "w");
        auto type = h5::make_datatype_for(data);
        auto space = h5::Dataspace::scalar();
        auto dset = file.require_dataset("data", type, space);

        THEN("The compound type can be written to a scalar dataset")
        {
            REQUIRE_NOTHROW(file.require_dataset("data", type, space));
            REQUIRE_NOTHROW(dset.write(data));
            REQUIRE(dset.read<hdf5_compound_test_t>() == data);
            REQUIRE_THROWS(dset.read<int>());
        }
        mara::filesystem::remove_file("test.h5");
    }
}

TEST_CASE("trees can be written to HDF5", "[arithmetic_binary_tree]")
{
    SECTION("tree indexes are formatted and read back correctly")
    {
        REQUIRE(mara::format_tree_index(mara::make_tree_index(0, 0, 0).with_level(0))   == "0:0-0-0");
        REQUIRE(mara::format_tree_index(mara::make_tree_index(5, 6, 7).with_level(3))   == "3:5-6-7");
        REQUIRE(mara::format_tree_index(mara::make_tree_index(1, 16, 17).with_level(5)) == "5:01-16-17");
        REQUIRE(mara::format_tree_index(mara::make_tree_index(1, 2).with_level(8))      == "8:001-002");

        REQUIRE(mara::read_tree_index<3>("8:000-000-000") == mara::make_tree_index(0, 0, 0).with_level(8));
        REQUIRE(mara::read_tree_index<3>("8:001-002-255") == mara::make_tree_index(1, 2, 255).with_level(8));
        REQUIRE_THROWS(mara::read_tree_index<2>("8:000-000-000"));
    }

    GIVEN("A tree with 4 levels")
    {
        auto bif = [] (auto&& c) { return mara::iota<4>(); };
        auto tree = mara::tree_of<2>(0)
        .bifurcate_all(bif)
        .bifurcate_all(bif)
        .bifurcate_all(bif)
        .bifurcate_all(bif);

        WHEN("The tree is written to a file")
        {
            auto file = h5::File("test.h5", "w");
            mara::write_tree(file.open_group("."), tree);
            file.close();

            WHEN("It is read back again")
            {
                auto file_r = h5::File("test.h5", "r");
                auto tree_r = decltype(tree)();
                mara::read_tree(file_r.open_group("."), tree_r);

                THEN("The tree has the correct size and values")
                {
                    REQUIRE(tree_r.size() == tree.size());
                    REQUIRE((tree_r == tree).all());
                }
            }
            mara::filesystem::remove_file("test.h5");
        }
    }
}

TEST_CASE("lists can be written to HDF5", "[linked_list]")
{
    GIVEN("A file opened for writing")
    {
        auto file = h5::File("test.h5", "w");

        WHEN("A list is written to the array")
        {
            auto a = mara::linked_list_t<double>().append(1.0).append(2.0);
            file.write("a", a);

            THEN("It can be read back again")
            {
                REQUIRE(file.read<decltype(a)>("a") == a);
            }
        }
        mara::filesystem::remove_file("test.h5");
    }
}

#endif // MARA_COMPILE_SUBPROGRAM_TEST
