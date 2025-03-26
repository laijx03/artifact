# BUILD Introduce

## Build Options
- Default Build Options Setting

    Global options
    ```
    -DCMAKE_BUILD_TYPE=Debug|Release    # default Debug
    -DBUILD_STATIC=ON|OFF               # default ON
    -DBUILD_SHARED=ON|OFF               # default OFF   TODO: implement later
    -DBUILD_UNITTEST=ON|OFF             # default ON
    -DBUILD_BENCH=ON|OFF                # default ON
    -DBUILD_DOC=ON|OFF                  # default OFF
    ```

    Current repo options
    ```
    -DNN_CODE_CHECK=ON|OFF              # default ON    NOTE: if want close code check, Manually set to OFF
    -DNN_WITH_SRC="air-infra"           # default OFF   NOTE: Need to install the air-infra.tar.gz package
    -DNN_BUILD_TEST=ON|OFF              # default ON
    -DNN_BUILD_EXAMPLE=ON|OFF           # default ON
    -DNN_INSTALL_APP=ON|OFF             # default ON
    ```

## Build

- Example: Build nn-addon with "air-infra" source


    Clone repos(air-infra & nn-addon) manually

    ```
    mkdir workarea && cd workarea
    git clone git@xxxx.xxxx.xxxx:air-infra/nn-addon.git --recurse
    git clone git@xxxx.xxxx.xxxx:air-infra/air-infra.git --recurse
    ```
    ```
    cmake -S nn-addon -B build -DNN_WITH_SRC="air-infra"
    cmake --build build -- -j"N"
    ```

- Example: Build nn-addon with "air-infra" libs

    Install libair_infra.a

    Daily builds of dependence libraries on ACI, include debug and release versions.
    The storage path is: oss://antsys-fhe/daily/***"DATE"***/***air-infra_deb/rel.tar.gz***
    Example: Get the dependency library and install, such as:
    ```
    ossutil64 cp oss://antsys-fhe/daily/20230718/air-infra_deb.tar.gz .
    tar xf air-infra_deb.tar.gz -C /usr/local
    ```
    ```
    cmake -S nn-addon -B build
    cmake --build build -- -j"N"
    ```

- Example: Build nn-addon with Debug

    ```
    cmake -S nn-addon -B build -DCMAKE_BUILD_TYPE=Debug
    cmake --build build -- -j"N"
    ```

- Install

    ```
    cmake --install build
    ```
