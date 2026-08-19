


#ifndef lonexgrad_STORAGE_BASE_H_
#define lonexgrad_STORAGE_BASE_H_

namespace lonexgrad {


class StorageBase {
public:
    StorageBase() = default;
    StorageBase(const StorageBase &) = default;
    StorageBase(StorageBase &&) = default;
    StorageBase &operator=(const StorageBase &) = default;
    StorageBase &operator=(StorageBase &&) = default;
    virtual ~StorageBase() = default;
};

}    

#endif    
