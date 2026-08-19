target_sources(lonexgrad PRIVATE
    lonex_binary.h
    lonex_dispatch.h
    lonex_distribution.h
    lonex_reduce.h
    lonex_span.h
    lonex_unary.h
    lonex_util.h

    kernel/binary.hpp
    kernel/cast.hpp
    kernel/distribution.hpp
    kernel/reduce.hpp
    kernel/unary.hpp
)
