# `srbset`

`srbset` is a Rust library providing a performant, memory-efficient sparse set data structure for `usize` integers, leveraging the capabilities of `roaring::RoaringBitmap` and segmenting to handle indices beyond `u32::MAX`.

This crate is ideal for scenarios where you need to store and manipulate very large sets of integers, especially when the data is sparse or spans across large ranges, potentially exceeding the 32-bit integer limit.

## Features

* **Efficient Sparse Storage:** Utilizes `RoaringBitmap`s for optimal memory usage when storing sparse sets of integers.
* **Large Index Support:** Seamlessly handles `usize` indices, segmenting the underlying `RoaringBitmap`s to support values well beyond `u32::MAX`.
* **Common Set Operations:** Provides intuitive `std::ops` implementations for:
    * Union (`|`)
    * Intersection (`&`)
    * Difference (`-`)
* **Iteration:** Iterate over all elements in the set in ascending order.
* **Containment Check:** Efficiently check if an element is present in the set.
* **Subset Check:** Determine if one set is a subset of another.
* **Length and Emptiness:** Query the number of elements and check if the set is empty.
* **Easy Construction:** Create sets directly from iterators of `usize`.

## License

Dual-licensed to be compatible with the Rust project.

Licensed under the Apache License, Version 2.0 http://www.apache.org/licenses/LICENSE-2.0 or the MIT license http://opensource.org/licenses/MIT, at your option. This file may not be copied, modified, or distributed except according to those terms.