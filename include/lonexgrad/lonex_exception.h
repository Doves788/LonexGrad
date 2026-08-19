


#ifndef lonexgrad_EXCEPTION_H_
#define lonexgrad_EXCEPTION_H_

#include <lonexgrad/export.h>

#include <exception>
#include <iostream>
#include <string>

namespace lonexgrad {

class LONEXGRAD_EXPORT TTException : public std::exception {
public:
    TTException(const std::string &message, const char *file_name, const char *function_signature, int line_number);
    [[nodiscard]] auto what() const noexcept -> const char * override;

private:
    std::string message;
};

// NOLINTNEXTLINE(*-macro-usage,*-pro-bounds-array-to-pointer-decay)
#define LONEXGRAD_EXCEPTION(msg) throw TTException(msg, __FILE__, __PRETTY_FUNCTION__, __LINE__)

// NOLINTNEXTLINE(*-macro-usage,*-pro-bounds-array-to-pointer-decay)
#define LONEXGRAD_ERROR(msg)                                                        \
    {                                                                        \
        try {                                                                \
            throw TTException(msg, __FILE__, __PRETTY_FUNCTION__, __LINE__); \
        } catch (const TTException &e) {                                     \
            std::cout << e.what() << std::endl;                              \
        }                                                                    \
        std::exit(1);                                                        \
    }

}    

#endif    
