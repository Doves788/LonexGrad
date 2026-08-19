


#ifndef lonexgrad_PRINT_OPTIONS_H_
#define lonexgrad_PRINT_OPTIONS_H_

#include <lonexgrad/export.h>

namespace lonexgrad {





LONEXGRAD_EXPORT void set_print_precision(int precision);





LONEXGRAD_EXPORT auto get_print_precision() -> int;





LONEXGRAD_EXPORT void set_print_width(int width);





LONEXGRAD_EXPORT auto get_print_width() -> int;





LONEXGRAD_EXPORT void set_max_lines(int max_lines);





LONEXGRAD_EXPORT auto get_max_lines() -> int;





LONEXGRAD_EXPORT void set_print_line_width(int line_width);





LONEXGRAD_EXPORT auto get_print_line_width() -> int;





LONEXGRAD_EXPORT void set_print_suppression(bool suppress);





LONEXGRAD_EXPORT auto get_print_suppression() -> bool;

}    

#endif    
