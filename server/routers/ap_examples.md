# API Call Examples for Progressions Router

Assuming the FastAPI server is running locally on port 8000.

## Test Endpoint

```bash
curl -X GET "http://127.0.0.1:8000/progressions/test" -H "accept: application/json"
```

## Arithmetic Progression (AP)

### Basic AP Calculation (nth term and sum)

```bash
curl -X POST "http://127.0.0.1:8000/progressions/ap/basic" \
-H "accept: application/json" \
-H "Content-Type: application/json" \
-d '{
  "a": 2,
  "d": 3,
  "n": 5
}'
```

_(Calculates the 5th term and sum of the first 5 terms for AP starting with 2, common difference 3)_

### Nth Term from Last (AP)

```bash
curl -X POST "http://127.0.0.1:8000/progressions/ap/nth_term_from_last" \
-H "accept: application/json" \
-H "Content-Type: application/json" \
-d '{
  "d": 4,
  "l": 43,
  "n": 3
}'
```

_(Calculates the 3rd term from the end of an AP with common difference 4 and last term 43)_

### Middle Term(s) (AP)

```bash
curl -X POST "http://127.0.0.1:8000/progressions/ap/middle_term" \
-H "accept: application/json" \
-H "Content-Type: application/json" \
-d '{
  "a": 5,
  "d": 2,
  "last_term": 23
}'
```

_(Finds the middle term(s) of the AP starting at 5, common difference 2, ending at 23)_

## Geometric Progression (GP)

### Basic GP Calculation (nth term and sum)

```bash
curl -X POST "http://127.0.0.1:8000/progressions/gp/basic" \
-H "accept: application/json" \
-H "Content-Type: application/json" \
-d '{
  "a": 3,
  "r": 2,
  "n": 4
}'
```

_(Calculates the 4th term and sum of the first 4 terms for GP starting with 3, common ratio 2)_

### Sum to Infinity (GP)

```bash
curl -X POST "http://127.0.0.1:8000/progressions/gp/sum_infinity" \
-H "accept: application/json" \
-H "Content-Type: application/json" \
-d '{
  "a": 10,
  "r": 0.5
}'
```

_(Calculates the sum to infinity for GP starting with 10, common ratio 0.5)_

### Geometric Mean

```bash
curl -X POST "http://127.0.0.1:8000/progressions/gp/geometric_mean" \
-H "accept: application/json" \
-H "Content-Type: application/json" \
-d '{
  "num1": 4,
  "num2": 9
}'
```

_(Calculates the geometric mean of 4 and 9)_

### Insert Geometric Means

```bash
curl -X POST "http://127.0.0.1:8000/progressions/gp/insert_means" \
-H "accept: application/json" \
-H "Content-Type: application/json" \
-d '{
  "a": 2,
  "b": 162,
  "k": 3
}'
```

_(Inserts 3 geometric means between 2 and 162)_
