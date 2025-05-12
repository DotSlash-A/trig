## How to Use Vectors in C++

Vectors in C++ are dynamic arrays that can grow or shrink in size. They are part of the Standard Template Library (STL) and offer many useful functions for initialization, modification, and access.

---

**Initialization**

- **Empty Vector:**
  ```cpp
  std::vector v; // creates an empty vector of integers
  ```
- **With Size and Value:**
  ```cpp
  std::vector v(5, 2); // creates a vector of size 5, all elements initialized to 2
  ```
- **Using Initializer List:**
  ```cpp
  std::vector v = {10, 20, 30}; // creates a vector with elements 10, 20, 30
  ```
- **From Another Vector or Range:**
  ```cpp
  std::vector v1 = {1, 2, 3, 4, 5};
  std::vector v2(v1.begin(), v1.begin() + 3); // v2 = {1, 2, 3}
  ```
- **Using fill():**
  ```cpp
  std::vector v(5);
  std::fill(v.begin(), v.end(), 6); // v = {6, 6, 6, 6, 6}
  ```

---

**Modification**

- **Add Element to End:**
  ```cpp
  v.push_back(42); // adds 42 at the end
  ```
- **Remove Last Element:**
  ```cpp
  v.pop_back(); // removes the last element
  ```
- **Insert at Position:**
  ```cpp
  v.insert(v.begin() + 1, 99); // inserts 99 at index 1
  ```
- **Erase Element(s):**
  ```cpp
  v.erase(v.begin() + 2); // removes element at index 2
  v.erase(v.begin(), v.begin() + 2); // removes first two elements
  ```
- **Change Value:**
  ```cpp
  v[0] = 100; // sets first element to 100
  v.at(1) = 200; // sets second element to 200 (with bounds checking)
  ```
- **Clear All Elements:**
  ```cpp
  v.clear(); // removes all elements, size becomes 0
  ```

---

**Accessing Elements**

- **By Index:**
  ```cpp
  int x = v[2]; // third element (no bounds checking)
  int y = v.at(2); // third element (with bounds checking)
  ```
- **First and Last Element:**
  ```cpp
  int first = v.front();
  int last = v.back();
  ```
- **Size of Vector:**
  ```cpp
  size_t sz = v.size();
  ```
- **Iterating:**
  ```cpp
  for (int val : v) {
      std::cout << val << " ";
  }
  ```

---

**Summary Table**

| Operation    | Function/Method | Example                    |
| ------------ | --------------- | -------------------------- | ------ |
| Add to end   | `push_back()`   | `v.push_back(5);`          |
| Remove last  | `pop_back()`    | `v.pop_back();`            |
| Insert       | `insert()`      | `v.insert(v.begin(), 10);` |
| Erase        | `erase()`       | `v.erase(v.begin());`      |
| Change value | `[]`, `.at()`   | `v=1; v.at(1)=2;`          |
| Access size  | `size()`        | `v.size();`                |
| Clear all    | `clear()`       | `v.clear();`               | [4][5] |

---

Vectors are powerful and flexible, making them the preferred choice for dynamic arrays in C++ programming[1][2][3][4][5].
