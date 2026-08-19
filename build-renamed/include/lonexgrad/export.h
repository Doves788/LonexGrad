
#ifndef LONEXGRAD_EXPORT_H
#define LONEXGRAD_EXPORT_H

#ifdef LONEXGRAD_STATIC_DEFINE
#  define LONEXGRAD_EXPORT
#  define LONEXGRAD_NO_EXPORT
#else
#  ifndef LONEXGRAD_EXPORT
#    ifdef lonexgrad_EXPORTS
        /* We are building this library */
#      define LONEXGRAD_EXPORT 
#    else
        /* We are using this library */
#      define LONEXGRAD_EXPORT 
#    endif
#  endif

#  ifndef LONEXGRAD_NO_EXPORT
#    define LONEXGRAD_NO_EXPORT 
#  endif
#endif

#ifndef LONEXGRAD_DEPRECATED
#  define LONEXGRAD_DEPRECATED __declspec(deprecated)
#endif

#ifndef LONEXGRAD_DEPRECATED_EXPORT
#  define LONEXGRAD_DEPRECATED_EXPORT LONEXGRAD_EXPORT LONEXGRAD_DEPRECATED
#endif

#ifndef LONEXGRAD_DEPRECATED_NO_EXPORT
#  define LONEXGRAD_DEPRECATED_NO_EXPORT LONEXGRAD_NO_EXPORT LONEXGRAD_DEPRECATED
#endif

/* NOLINTNEXTLINE(readability-avoid-unconditional-preprocessor-if) */
#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef LONEXGRAD_NO_DEPRECATED
#    define LONEXGRAD_NO_DEPRECATED
#  endif
#endif

#endif /* LONEXGRAD_EXPORT_H */
