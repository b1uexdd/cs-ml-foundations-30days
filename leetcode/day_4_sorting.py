#冒泡排序
def bubble_sort(nums):
    n = len(nums)

    for i in range(n):
        for j in range(0, n - 1 - i):
            if nums[j] > nums[j+1]:
                nums[j], nums[j+1] = nums[j+1], nums[j]
    
    return nums

#选择排序
def selection_sort(nums):
    n = len(nums)

    for i in range(n):
        min_idx = i

        for j in range(i+1, n):
            if nums[j] < nums[i]:
                min_idx = j
        
        nums[i], nums[min_idx] = nums[min_idx], nums[i]

    return nums

#插入排序
def insertion_sort(nums):
    n = len(nums)

    for i in range(1, n):
        key = nums[i]
        j = i - 1

        while j >= 0 and nums[j] > key:
            nums[j + 1] = nums[j]
            j -= 1

        nums[j + 1] = key

    return nums

if __name__ == "__main__":
    test_cases = [
        [5, 2, 3, 1],
        [1, 2, 3, 4],
        [4, 3, 2, 1],
        [2, 2, 1, 3],
        [],
        [1]
    ]

    for nums in test_cases:
        expected = sorted(nums)

        assert bubble_sort(nums.copy()) == expected
        assert selection_sort(nums.copy()) == expected
        assert insertion_sort(nums.copy()) == expected

    print("All tests passed!")