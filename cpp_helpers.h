#ifndef __CPP_HELPERS_H__
#define __CPP_HELPERS_H__

#define NON_COPYABLE( typename_t )                       \
    typename_t( const typename_t& ) = delete;            \
    typename_t& operator=( const typename_t& ) = delete

#endif // !__CPP_HELPERS_H__
