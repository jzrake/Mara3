



//=============================================================================
template<typename DerivedClass, typename... Types>
class covariant_tuple_t
{
public:

    //=========================================================================
    covariant_tuple_t() {}
    covariant_tuple_t(std::tuple<Types...> the_tuple) : the_tuple(the_tuple) {}

    DerivedClass operator+(const DerivedClass& other) const { return transform(other, std::plus<>()); }
    DerivedClass operator-(const DerivedClass& other) const { return transform(other, std::minus<>()); }

    template<std::size_t Index>
    const auto& get() const { return std::get<Index>(the_tuple); }

    template<typename Function>
    auto transform(Function&& fn) const
    {
        auto is = std::make_index_sequence<sizeof...(Types)>();
        return transform_impl(std::forward<Function>(fn), std::move(is));
    }

    template<typename Function>
    auto transform(const DerivedClass& other, Function&& fn) const
    {
        auto is = std::make_index_sequence<sizeof...(Types)>();
        return transform_impl(other, std::forward<Function>(fn), std::move(is));
    }

private:
    //=========================================================================
    template<typename Function, std::size_t... Is>
    auto transform_impl(Function&& fn, std::index_sequence<Is...>&&) const
    {
        return std::make_tuple(fn(std::get<Is>(the_tuple))...);
    }

    template<typename Function, std::size_t... Is>
    auto transform_impl(const DerivedClass& other, Function&& fn, std::index_sequence<Is...>&&) const
    {
        return std::make_tuple(fn(std::get<Is>(the_tuple), std::get<Is>(other.the_tuple))...);
    }

    std::tuple<Types...> the_tuple;
};

