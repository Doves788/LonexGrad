# Adding A Backend
To add a backend, the following must be done:

## Create Backend Device
In [device.h](../include/tt/device.h)
- Add a backend name to `enum class Backend`
- Handle the `to_string` case for your new backend
- Create a shortname `constexpr Device`

## Implement Backend
In [lonexgrad/tensor/backend](../lonexgrad/tensor/backend)
- Create a folder for all your implementation files
- Your backend must derive and extend [BackendBase](../lonexgrad/tensor/backend_base.h)

## Register The Backend
Add your device-to-backend in [get_backend](../lonexgrad/tensor/backend_register.cpp)

## Register Backend For Testing
Add your backend device in [runner_single_type](../tests/test_util.h)
