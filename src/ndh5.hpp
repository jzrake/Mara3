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




#pragma once
#include <string>
#include <vector>
#include <stdexcept>
#include <hdf5.h>




//=============================================================================
namespace h5
{
    template <class GroupType, class DatasetType>
    class Location;
    class PropertyList;
    class Link;
    class File;
    class Group;
    class Dataset;
    class Datatype;
    class Dataspace;
    struct hyperslab_t;

    enum class Intent { rdwr, rdonly, swmr_write, swmr_read };
    enum class Object { file, group, dataset };

    template<typename T> struct hdf5_type_info {};
    template<typename T> static auto make_datatype_for(const T&);
    template<typename T> static auto make_dataspace_for(const T&);
    template<typename T> static auto prepare(const Datatype&, const Dataspace&);
    template<typename T> static auto get_address(T&);
    template<typename T> static auto get_address(const T&);

    namespace detail
    {
        static inline herr_t get_last_error(unsigned, const H5E_error2_t*, void*);
        template<typename T> static T check(T);
    }
}




//=============================================================================
herr_t h5::detail::get_last_error(unsigned n, const H5E_error2_t *err, void *data)
{
    if (n == 0)
    {
        *static_cast<H5E_error2_t*>(data) = *err;
    }
    return 0;
}

template<typename T>
T h5::detail::check(T result)
{
    if (result < 0)
    {
        H5E_error2_t err;
        hid_t eid = H5Eget_current_stack();
        H5Ewalk(eid, H5E_WALK_UPWARD, get_last_error, &err);
        std::string what = err.desc;
        H5Eclear(eid);
        H5Eclose_stack(eid);
        throw std::invalid_argument(what);
    }
    return result;
}




// ============================================================================
struct h5::hyperslab_t
{
    void check_valid(hsize_t rank) const
    {
        if (start.size() != rank ||
            skips.size() != rank ||
            count.size() != rank ||
            block.size() != rank)
        {
            throw std::invalid_argument("inconsistent selection sizes");
        }
    }

    std::vector<hsize_t> start;
    std::vector<hsize_t> count;
    std::vector<hsize_t> skips;
    std::vector<hsize_t> block;
};




//=============================================================================
class h5::PropertyList final
{
public:
    static PropertyList dataset_create() { return H5Pcreate(H5P_DATASET_CREATE); }

    ~PropertyList() { close(); }
    PropertyList(const PropertyList& other) : id(H5Pcopy(other.id)) {}
    PropertyList(PropertyList&& other)
    {
        id = other.id;
        other.id = -1;
    }

    void close()
    {
        if (id != -1)
        {
            H5Pclose(id);
        }
    }

    PropertyList& set_chunk(std::vector<hsize_t> dims)
    {
        auto hdims = std::vector<hsize_t>(dims.begin(), dims.end());
        detail::check(H5Pset_chunk(id, int(hdims.size()), &hdims[0]));
        return *this;
    }

    template<typename... Args>
    PropertyList& set_chunk(Args... args)
    {
        return set_chunk(std::vector<hsize_t>{hsize_t(args)...});
    }

    bool operator==(const PropertyList& other) const
    {
        return detail::check(H5Pequal(id, other.id));
    }

private:
    //=========================================================================
    friend class Dataset;
    friend class Link;

    PropertyList(hid_t id) : id(id) {}
    hid_t id = -1;
};




//=============================================================================
class h5::Datatype final
{
public:

    //=========================================================================
    static Datatype native_double() { return H5Tcopy(H5T_NATIVE_DOUBLE); }
    static Datatype native_int()    { return H5Tcopy(H5T_NATIVE_INT); }
    static Datatype c_s1()          { return H5Tcopy(H5T_C_S1); }

    //=========================================================================
    ~Datatype() { close(); }
    Datatype() {}
    Datatype(const Datatype& other) : id(H5Tcopy(other.id)) {}
    Datatype(Datatype&& other)
    {
        id = other.id;
        other.id = -1;
    }

    void close()
    {
        if (id != -1)
        {
            H5Tclose(id);
        }
    }

    Datatype& operator=(const Datatype& other)
    {
        close();
        id = detail::check(H5Tcopy(other.id));
        return *this;
    }

    bool operator==(const Datatype& other) const
    {
        return detail::check(H5Tequal(id, other.id));
    }

    bool operator!=(const Datatype& other) const
    {
        return ! operator==(other);
    }

    std::size_t size() const
    {
        return detail::check(H5Tget_size(id));
    }

    Datatype with_size(std::size_t size) const
    {
        Datatype other = *this;
        H5Tset_size(other.id, size);
        return other;
    }

    Datatype as_array(std::size_t size) const
    {
        hsize_t dims = size;
        return detail::check(H5Tarray_create(id, 1, &dims));
    }

private:
    //=========================================================================
    friend class Link;
    friend class Dataset;

    Datatype(hid_t id) : id(id) {}
    hid_t id = -1;
};




//=============================================================================
class h5::Dataspace
{
public:

    //=========================================================================
    static Dataspace scalar()
    {
        return detail::check(H5Screate(H5S_SCALAR));
    }

    template<typename Container>
    static Dataspace simple(Container dims)
    {
        auto hdims = std::vector<hsize_t>(dims.begin(), dims.end());
        return detail::check(H5Screate_simple(int(hdims.size()), &hdims[0], nullptr));
    }

    template<typename Container>
    static Dataspace simple(Container dims, Container max_dims)
    {
        if (dims.size() != max_dims.size())
        {
            throw std::invalid_argument("dims and max dims sizes do not agree");
        }
        auto hdims = std::vector<hsize_t>(dims.begin(), dims.end());
        auto max_hdims = std::vector<hsize_t>(max_dims.begin(), max_dims.end());
        return detail::check(H5Screate_simple(int(hdims.size()), &hdims[0], &max_hdims[0]));
    }

    template<typename... Args>
    static Dataspace unlimited(Args... initial_sizes)
    {
        auto initial = std::vector<hsize_t> {hsize_t(initial_sizes)...};
        auto maximum = std::vector<hsize_t>(sizeof...(initial_sizes), H5S_UNLIMITED);
        return simple(initial, maximum);
    }

    //=========================================================================
    Dataspace() : id (detail::check(H5Screate(H5S_NULL))) {}
    Dataspace(const Dataspace& other) : id(detail::check(H5Scopy(other.id))) {}
    Dataspace(std::initializer_list<std::size_t> dims) : Dataspace(dims.size() == 0 ? scalar() : simple(std::vector<size_t>(dims))) {}
    ~Dataspace() { close(); }

    Dataspace& operator=(const Dataspace& other)
    {
        close();
        id = detail::check(H5Scopy(other.id));
        return *this;
    }

    bool operator==(const Dataspace& other) const
    {
        return detail::check(H5Sextent_equal(id, other.id));
    }

    bool operator!=(const Dataspace& other) const
    {
        return ! operator==(other);
    }

    void close()
    {
        if (id != -1)
        {
            detail::check(H5Sclose(id));
        }
    }

    std::size_t rank() const
    {
        return detail::check(H5Sget_simple_extent_ndims(id));
    }

    std::size_t size() const
    {
        return detail::check(H5Sget_simple_extent_npoints(id));
    }

    std::vector<std::size_t> extent() const
    {
        auto ext = std::vector<hsize_t>(rank());
        detail::check(H5Sget_simple_extent_dims(id, &ext[0], nullptr));
        return std::vector<std::size_t>(ext.begin(), ext.end());
    }

    std::size_t selection_size() const
    {
        return detail::check(H5Sget_select_npoints(id));
    }

    std::vector<std::size_t> selection_lower() const
    {
        auto lower = std::vector<hsize_t>(rank());
        auto upper = std::vector<hsize_t>(rank());
        detail::check(H5Sget_select_bounds(id, &lower[0], &upper[0]));
        return std::vector<std::size_t>(lower.begin(), lower.end());
    }

    std::vector<std::size_t> selection_upper() const
    {
        auto lower = std::vector<hsize_t>(rank());
        auto upper = std::vector<hsize_t>(rank());
        detail::check(H5Sget_select_bounds(id, &lower[0], &upper[0]));
        return std::vector<std::size_t>(upper.begin(), upper.end());
    }

    Dataspace& select_all()
    {
        detail::check(H5Sselect_all(id));
        return *this;
    }

    Dataspace& select_none()
    {
        detail::check(H5Sselect_none(id));
        return *this;
    }

    Dataspace& select(const hyperslab_t& selection)
    {
        selection.check_valid(rank());
        detail::check(H5Sselect_hyperslab(id, H5S_SELECT_SET,
            selection.start.data(),
            selection.skips.data(),
            selection.count.data(),
            selection.block.data()));
        return *this;
    }

private:
    //=========================================================================
    friend class Link;
    friend class Dataset;

    Dataspace(hid_t id) : id(id) {}
    hid_t id = -1;
};




//=============================================================================
template<>
struct h5::hdf5_type_info<char>
{
    using native_type = char;
    static auto make_datatype_for(const native_type&) { return Datatype::c_s1(); }
    static auto make_dataspace_for(const native_type&) { return Dataspace::scalar(); }
    static auto prepare(const Datatype&, const Dataspace&) { return native_type(); }
    static auto finalize(native_type&& value) { return std::move(value); }
    static auto get_address(native_type& value) { return &value; }
    static auto get_address(const native_type& value) { return &value; }
};

template<>
struct h5::hdf5_type_info<int>
{
    using native_type = int;
    static Datatype make_datatype_for(const native_type&) { return Datatype::native_int(); }
    static Dataspace make_dataspace_for(const native_type&) { return Dataspace::scalar(); }
    static native_type prepare(const Datatype&, const Dataspace&) { return native_type(); }
    static auto finalize(native_type&& value) { return std::move(value); }
    static void* get_address(native_type& value) { return &value; }
    static const void* get_address(const native_type& value) { return &value; }
};

template<>
struct h5::hdf5_type_info<double>
{
    using native_type = double;
    static auto make_datatype_for(const native_type&) { return Datatype::native_double(); }
    static auto make_dataspace_for(const native_type&) { return Dataspace::scalar(); }
    static auto prepare(const Datatype&, const Dataspace&) { return native_type(); }
    static auto finalize(native_type&& value) { return std::move(value); }
    static auto get_address(native_type& value) { return &value; }
    static auto get_address(const native_type& value) { return &value; }
};

template<>
struct h5::hdf5_type_info<std::string>
{
    using native_type = std::string;
    static auto make_datatype_for(const native_type& value) { return Datatype::c_s1().with_size(std::max(std::size_t(1), value.size())); }
    static auto make_dataspace_for(const native_type&) { return Dataspace::scalar(); }
    static auto prepare(const Datatype& type, const Dataspace&) { return native_type(type.size(), 0); }
    static auto finalize(native_type&& value) { return std::move(value); }
    static auto get_address(native_type& value) { return value.data(); }
    static auto get_address(const native_type& value) { return value.data(); }
};

template<typename ValueType>
struct h5::hdf5_type_info<std::vector<ValueType>>
{
    using native_type = std::vector<ValueType>;
    static auto make_datatype_for(const native_type&) { return h5::make_datatype_for(ValueType()); }
    static auto make_dataspace_for(const native_type& value) { return Dataspace{value.size()}; }
    static auto prepare(const Datatype&, const Dataspace& space) { return native_type(space.size()); }
    static auto finalize(native_type&& value) { return std::move(value); }
    static auto get_address(native_type& value) { return value.data(); }
    static auto get_address(const native_type& value) { return value.data(); }
};




//=============================================================================
template<typename T> auto h5::make_datatype_for(const T& v) { return hdf5_type_info<T>::make_datatype_for(v); }
template<typename T> auto h5::make_dataspace_for(const T& v) { return hdf5_type_info<T>::make_dataspace_for(v); }
template<typename T> auto h5::prepare(const Datatype& t, const Dataspace& s) { return hdf5_type_info<T>::prepare(t, s); }
template<typename T> auto h5::get_address(T& v) { return hdf5_type_info<T>::get_address(v); }
template<typename T> auto h5::get_address(const T& v) { return hdf5_type_info<T>::get_address(v); }




//=============================================================================
class h5::Link
{
private:

    //=========================================================================
    ~Link() { /*assert(id == -1);*/ } // link must be closed before going out of scope
    Link() {}
    Link(hid_t id) : id(id) {}
    Link(Link&& other)
    {
        id = other.id;
        other.id = -1;
    }
    Link(const Link& other) = delete;

    Link& operator=(Link&& other)
    {
        id = other.id;
        other.id = -1;
        return *this;
    }

    void close(Object object)
    {
        if (id != -1)
        {
            switch (object)
            {
                case Object::file   : H5Fclose(id); break;
                case Object::group  : H5Gclose(id); break;
                case Object::dataset: H5Dclose(id); break;
            }
            id = -1;
        }
    }

    std::size_t size() const
    {
        auto op = [] (auto, auto, auto, auto) { return 0; };
        auto idx = hsize_t(0);
        H5Literate(id, H5_INDEX_NAME, H5_ITER_INC, &idx, op, nullptr);
        return idx;
    }

    bool contains(const std::string& name, Object object) const
    {
        if (H5Lexists(id, name.data(), H5P_DEFAULT))
        {
            H5O_info_t info;
            H5Oget_info_by_name(id, name.data(), &info, H5P_DEFAULT);

            switch (object)
            {
                case Object::file   : return false;
                case Object::group  : return info.type == H5O_TYPE_GROUP;
                case Object::dataset: return info.type == H5O_TYPE_DATASET;
            }
        }
        return false;
    }

    Link open_group(const std::string& name)
    {
        return detail::check(H5Gopen(id, name.data(),
            H5P_DEFAULT));
    }

    Link create_group(const std::string& name)
    {
        return detail::check(H5Gcreate(id, name.data(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    }

    Link open_dataset(const std::string& name)
    {
        return detail::check(H5Dopen(id, name.data(), H5P_DEFAULT));
    }

    Link create_dataset(const std::string& name,
                        const Datatype& type,
                        const Dataspace& space,
                        const PropertyList& creation_plist=PropertyList::dataset_create())
    {
        return detail::check(H5Dcreate(
            id,
            name.data(),
            type.id,
            space.id,
            H5P_DEFAULT,
            creation_plist.id,
            H5P_DEFAULT));
    }

    //=========================================================================
    class iterator
    {
    public:

        using value_type = std::string;
        using iterator_category = std::forward_iterator_tag;

        //=====================================================================
        iterator(hid_t id, hsize_t idx) : id(id), idx(idx) {}
        iterator& operator++() { ++idx; return *this; }
        iterator operator++(int) { auto ret = *this; this->operator++(); return ret; }
        bool operator==(iterator other) const { return id == other.id && idx == other.idx; }
        bool operator!=(iterator other) const { return id != other.id || idx != other.idx; }
        std::string operator*() const
        {
            static char name[1024];

            if (H5Lget_name_by_idx(id, ".",
                H5_INDEX_NAME, H5_ITER_NATIVE, idx, name, 1024, H5P_DEFAULT) > 1024)
            {
                throw std::overflow_error("object names longer than 1024 are not supported");
            }
            return name;
        }

    private:
        //=====================================================================
        hid_t id = -1;
        hsize_t idx = 0;
    };

    iterator begin() const
    {
        return iterator(id, 0);
    }

    iterator end() const
    {
        return iterator(id, size());
    }

    //=========================================================================
    template <class GroupType, class DatasetType>
    friend class Location;
    friend class File;
    friend class Group;
    friend class Dataset;

    hid_t id = -1;
};




//=============================================================================
class h5::Dataset final
{
public:

    //=========================================================================
    ~Dataset() { close(); }
    Dataset() {}
    Dataset(Dataset&& other) : link(std::move(other.link)) {}
    Dataset(const Dataset&) = delete;
    Dataset& operator=(Dataset&& other)
    {
        link = std::move(other.link);
        return *this;
    }
    void close() { link.close(Object::dataset); }

    Dataspace get_space() const
    {
        return detail::check(H5Dget_space(link.id));
    }

    Datatype get_type() const
    {
        return detail::check(H5Dget_type(link.id));
    }

    template<typename T>
    void write(const T& value)
    {
        write(value, get_space());
    }

    template<typename T>
    void write(const T& value, const Dataspace& fspace)
    {
        auto data = get_address(value);
        auto type = make_datatype_for(value);
        auto mspace = make_dataspace_for(value);
        check_compatible(type);
        detail::check(H5Dwrite(link.id, type.id, mspace.id, fspace.id, H5P_DEFAULT, data));
    }

    template<typename T>
    T read()
    {
        return read<T>(get_space());
    }

    template<typename T>
    T read(const Dataspace& fspace)
    {
        auto value = prepare<T>(get_type(), fspace);
        auto data = get_address(value);
        auto type = make_datatype_for(value);
        auto mspace = make_dataspace_for(value);
        check_compatible(type);
        detail::check(H5Dread(link.id, type.id, mspace.id, fspace.id, H5P_DEFAULT, data));
        return hdf5_type_info<T>::finalize(std::move(value));
    }

    template<typename T>
    void read(T& value, const Dataspace& fspace)
    {
        value = read<T>(fspace);
    }

    template<typename T>
    void read(T& value)
    {
        value = read<T>();
    }

    PropertyList get_creation_plist() const
    {
        return detail::check(H5Dget_create_plist(link.id));
    }

    Dataset& set_extent(const std::vector<hsize_t>& hdims)
    {
        if (get_space().rank() != hdims.size())
        {
            throw std::invalid_argument("new and old extents have different ranks");
        }
        detail::check(H5Dset_extent(link.id, &hdims[0]));
        return *this;
    }

    template<typename... Args>
    Dataset& set_extent(Args... args)
    {
        return set_extent(std::vector<hsize_t>{hsize_t(args)...});
    }

private:
    //=========================================================================
    Datatype check_compatible(const Datatype& type) const
    {
        if (type != get_type())
        {
            throw std::invalid_argument("source and target have different data types");
        }
        return type;
    }

    //=========================================================================
    template <class GroupType, class DatasetType>
    friend class Location;

    Dataset(Link link) : link(std::move(link)) {}
    Link link;
};




//=============================================================================
template <class GroupType, class DatasetType>
class h5::Location
{
public:

    //=========================================================================
    Location() {}
    Location(const Group&) = delete;
    Location(Location&& other) : link(std::move(other.link)) {}
    Location& operator=(Location&& other)
    {
        link = std::move(other.link);
        return *this;
    }

    bool is_open() const
    {
        return link.id != -1;
    }

    std::size_t size() const
    {
        return link.size();
    }

    Link::iterator begin() const
    {
        return link.begin();
    }

    Link::iterator end() const
    {
        return link.end();
    }

    GroupType operator[](const std::string& name)
    {
        return require_group(name);
    }

    GroupType open_group(const std::string& name)
    {
        return link.open_group(name);
    }

    GroupType require_group(const std::string& name)
    {
        if (link.contains(name, Object::group))
        {
            return open_group(name);
        }
        return link.create_group(name);
    }

    DatasetType open_dataset(const std::string& name)
    {
        return link.open_dataset(name);
    }

    DatasetType require_dataset(const std::string& name,
                                const Datatype& type,
                                const Dataspace& space,
                                const PropertyList& creation_plist=PropertyList::dataset_create())
    {
        if (link.contains(name, Object::dataset))
        {
            auto dset = open_dataset(name);

            if (dset.get_type() == type &&
                dset.get_space() == space &&
                dset.get_creation_plist() == PropertyList::dataset_create())
            {
                return dset;
            }
            throw std::invalid_argument(
                "data set with different type or space already exists");
        }
        return link.create_dataset(name, type, space, creation_plist);
    }

    template<typename T>
    DatasetType require_dataset(const std::string& name, const Dataspace& space={})
    {
        return require_dataset(name, hdf5_type_info<T>::make_datatype_for(T()), space);
    }

    template<typename T>
    void write(const std::string& name, const T& value)
    {
        auto type  = make_datatype_for(value);
        auto space = make_dataspace_for(value);
        require_dataset(name, type, space).write(value);
    }

    template<typename T>
    T read(const std::string& name)
    {
        return open_dataset(name).template read<T>();
    }

    template<typename T>
    void read(const std::string& name, T& value)
    {
        value = read<T>(name);
    }

protected:
    //=========================================================================
    Location(Link&& link) : link(std::move(link)) {}
    Link link;
};




//=============================================================================
class h5::Group final : public Location<Group, Dataset>
{
public:

    //=========================================================================
    ~Group() { close(); }
    Group() {}
    Group(const Group&) = delete;
    Group(Group&& other) : Location(std::move(other.link)) {}
    Group& operator=(Group&& other)
    {
        link = std::move(other.link);
        return *this;
    }
    void close() { link.close(Object::group); }

private:
    //=========================================================================
    template <class GroupType, class DatasetType> friend class h5::Location;
    Group(Link link) : Location(std::move(link)) {}
};




//=============================================================================
class h5::File final : public Location<Group, Dataset>
{
public:

    //=========================================================================
    static bool exists(const std::string& filename)
    {
        return H5Fis_hdf5(filename.data()) > 0;
    }

    //=========================================================================
    ~File() { close(); }
    File() {}
    File(const File&) = delete;
    File(File&& other) : Location(std::move(other.link)) {}
    File(std::string filename, std::string mode="r")
    {
        if (mode == "r")
        {
            link.id = detail::check(H5Fopen(filename.data(), H5F_ACC_RDONLY, H5P_DEFAULT));
        }
        else if (mode == "r+")
        {
            link.id = detail::check(H5Fopen(filename.data(), H5F_ACC_RDWR, H5P_DEFAULT));
        }
        else if (mode == "w")
        {
            link.id = detail::check(H5Fcreate(filename.data(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT));
        }
        else
        {
            throw std::invalid_argument("File mode must be r, r+, or w");
        }
    }
    void close() { link.close(Object::file); }

    File& operator=(File&& other)
    {
        link = std::move(other.link);
        return *this;
    }

    Intent intent() const
    {
        unsigned intent;
        detail::check(H5Fget_intent(link.id, &intent));

        if (intent == H5F_ACC_RDWR)       return Intent::rdwr;
        if (intent == H5F_ACC_RDONLY)     return Intent::rdonly;

        throw;
    }

    std::string filename() const
    {
        char result[2048];

        if (H5Fget_name(link.id, result, 2048) > 2048)
        {
            throw std::invalid_argument("the HDF5 filename is too long");
        }
        return result;
    }

private:
    //=========================================================================
    File(Link link) : Location(std::move(link)) {}
};
