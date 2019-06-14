

//=============================================================================
template<std::size_t Size> auto sequence_of();
template<std::size_t Size> auto refine_points();




//=============================================================================
template<std::size_t Rank>
auto prolong_shape(nd::shape_t<Rank> shape, std::size_t axis)
{
    shape[axis] = shape[axis] * 2 - 1;
    return shape;
}

template<std::size_t Rank>
auto coarsen_index_lower(nd::index_t<Rank> fi, std::size_t axis)
{
    auto c0 = fi;
    c0[axis] = fi[axis] / 2;
    return c0;
}

template<std::size_t Rank>
auto coarsen_index_upper(nd::index_t<Rank> fi, std::size_t axis)
{
    auto c1 = fi;
    c1[axis] = fi[axis] / 2 + (fi[axis] % 2 == 0 ? 0 : 1);
    return c1;
}

auto prolong_points(std::size_t axis)
{
    return [axis] (auto coarse)
    {
        if (coarse.rank() <= axis)
            throw std::invalid_argument("prolong_points: cannot prolong on axis greater than or eaual to rank");

        return nd::make_array([axis, coarse] (auto i)
        {
            return 0.5 * (
                coarse(coarsen_index_lower(i, axis)) +
                coarse(coarsen_index_upper(i, axis)));
        }, prolong_shape(coarse.shape(), axis));
    };
}

auto bisect_points(std::size_t axis)
{
    return [axis] (auto parent)
    {
        if (parent.rank() <= axis)
            throw std::invalid_argument("bisect_points: cannot bisect on axis greater than or eaual to rank");
        if (parent.shape(axis) % 2 == 0)
            throw std::invalid_argument("bisect_points: must have an odd number of points");
        auto h0 = parent | nd::select_axis(axis).from(0).to(parent.shape(axis) / 2 + 1);
        auto h1 = parent | nd::select_axis(axis).from(parent.shape(axis) / 2).to(0).from_the_end();
        return std::make_tuple(h0, h1);
    };
}

auto bisect_points_lower(std::size_t axis)
{
    return [axis] (auto parent)
    {
        return std::get<0>(parent | bisect_points(axis));
    };
}

auto bisect_points_upper(std::size_t axis)
{
    return [axis] (auto parent)
    {
        return std::get<1>(parent | bisect_points(axis));
    };
}




//=============================================================================
template<>
auto sequence_of<4>()
{
    return [] (auto&& value)
    {
        return mara::make_sequence(value, value, value, value);
    };
}

template<>
auto refine_points<4>()
{
    return [] (auto array)
    {
        return mara::make_sequence(
            array | prolong_points(0) | prolong_points(1) | bisect_points_lower(0) | bisect_points_lower(1),
            array | prolong_points(0) | prolong_points(1) | bisect_points_lower(0) | bisect_points_upper(1),
            array | prolong_points(0) | prolong_points(1) | bisect_points_upper(0) | bisect_points_lower(1),
            array | prolong_points(0) | prolong_points(1) | bisect_points_upper(0) | bisect_points_upper(1));
    };
}



