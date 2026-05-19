"""
Day 3 - LeetCode 977：有序数组的平方

题目：
给定一个按非递减顺序排列的整数数组 nums，
返回每个数字的平方组成的新数组，并且新数组也要按非递减顺序排列。

例子：
输入：nums = [-4, -1, 0, 3, 10]
输出：[0, 1, 9, 16, 100]

核心思路：
原数组虽然是有序的，但是平方之后顺序可能会被打乱。
原因是负数平方以后会变成正数，而且绝对值越大的负数，平方后越大。

例如：
nums = [-4, -1, 0, 3, 10]
平方后变成：[16, 1, 0, 9, 100]

观察可以发现：
平方后的最大值一定来自数组的两端：
1. 最左边的数
2. 最右边的数

所以可以使用双指针：
- left 指向数组最左边
- right 指向数组最右边
- pos 指向结果数组当前要填的位置

为什么从后往前填？
因为每次比较 nums[left] 和 nums[right] 的绝对值时，
我们找到的是当前最大的平方数。
所以应该把它放到结果数组的最后面。
"""


class Solution(object):
    def sortedSquares(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """

        n = len(nums)
        res = [0] * n

        left = 0
        right = n - 1
        pos = n - 1

        while left <= right:
            if abs(nums[left]) > abs(nums[right]):
                res[pos] = nums[left] ** 2
                left += 1
            else:
                res[pos] = nums[right] ** 2
                right -= 1

            pos -= 1

        return res


"""
时间复杂度：
O(n)

原因：
数组中的每个元素只会被处理一次。

空间复杂度：
O(n)

原因：
我们创建了一个新的结果数组 res 来存放答案。

我学到的东西：
1. 一个有序数组在平方以后，不一定仍然有序。
2. 对于有序数组来说，平方后的最大值一定出现在数组两端。
3. 双指针可以把排序方法的 O(n log n) 优化到 O(n)。
4. 当每次先找到最大值时，应该从结果数组的后面往前填。
"""