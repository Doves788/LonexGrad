


#ifndef lonexgrad_SHAPE_H_
#define lonexgrad_SHAPE_H_

#include <lonexgrad/export.h>

#include <format>
#include <initializer_list>
#include <iostream>
#include <string>
#include <vector>

namespace lonexgrad {




class LONEXGRAD_EXPORT Shape {
public:
    Shape() = default;
    explicit Shape(const std::vector<int> &dims);
    explicit Shape(std::vector<int> &&dims);
    Shape(std::initializer_list<int> dims);

    



    [[nodiscard]] auto numel() const noexcept -> int;

    



    [[nodiscard]] auto ndim() const noexcept -> int;

    



    [[nodiscard]] auto size() const noexcept -> int;

    

    





    auto operator[](int index) -> int &;

    





    auto operator[](int index) const -> int;

    

    



    auto operator==(const Shape &other) const noexcept -> bool;

    



    auto operator!=(const Shape &other) const noexcept -> bool;

    



    auto operator==(const std::initializer_list<int> &other) const noexcept -> bool;

    



    auto operator!=(const std::initializer_list<int> &other) const noexcept -> bool;

    




    auto pop(int dim = -1) -> int;

    




    void insert(int size, int dim = -1);

    



    [[nodiscard]] auto to_string() const -> std::string;

    



    [[nodiscard]] auto to_stride() const -> Shape;

    



    [[nodiscard]] auto to_vec() const -> const std::vector<int> &;

private:
    std::vector<int> dims;
};

LONEXGRAD_EXPORT auto operator<<(std::ostream &os, const Shape &shape) -> std::ostream &;

LONEXGRAD_EXPORT auto to_string(const Shape &shape) -> std::string;

}    

template <>
struct LONEXGRAD_EXPORT std::formatter<lonexgrad::Shape> : std::formatter<std::string> {
    auto format(const lonexgrad::Shape &shape, format_context &ctx) const {
        return formatter<string>::format(std::format("{}", to_string(shape)), ctx);
    }
};

#endif    
