


#ifndef lonexgrad_DATAST_MNIST_H_
#define lonexgrad_DATAST_MNIST_H_

#include <lonexgrad/export.h>
#include <lonexgrad/lonex_shape.h>
#include <lonexgrad/lonex_tensor.h>

#include <string>
#include <tuple>

namespace lonexgrad::data {



class LONEXGRAD_EXPORT MNISTDataset {
public:
    using DataType = std::tuple<Tensor, Tensor>;

    





    MNISTDataset(const std::string &img_path, const std::string &label_path, bool normalize = true);

    


    [[nodiscard]] auto img_shape() const -> Shape;

    


    [[nodiscard]] auto size() const -> int;

    



    [[nodiscard]] auto get(int idx) const -> DataType;

private:
    MNISTDataset(const std::tuple<Tensor, Tensor, int> &);
    Tensor images;
    Tensor labels;
    int N;
};

}    

#endif    
