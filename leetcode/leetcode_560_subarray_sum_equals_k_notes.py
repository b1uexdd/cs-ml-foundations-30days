"""
LeetCode 560. Subarray Sum Equals K
题目：和为 K 的子数组

Goal / 目标:
Given an integer array nums and an integer k, return the total number of continuous subarrays
whose sum equals k.

给定一个整数数组 nums 和一个整数 k，返回数组中“连续子数组和等于 k”的个数。

Example / 示例:
nums = [1, 1, 1], k = 2

Valid subarrays:
[1, 1] from index 0 to 1
[1, 1] from index 1 to 2

Answer = 2
"""


class Solution(object):
    def subarraySum(self, nums, k):
        """
        Optimized solution: Prefix Sum + HashMap
        优化解法：前缀和 + 哈希表

        :type nums: List[int]
        :type k: int
        :rtype: int
        """

        # count records how many valid subarrays have sum equal to k.
        # count 记录当前找到多少个“和为 k 的连续子数组”。
        count = 0

        # cur_sum is the current prefix sum.
        # cur_sum 表示当前从 nums[0] 加到当前位置的总和。
        cur_sum = 0

        # seen records how many times each prefix sum has appeared before.
        # seen 记录“以前出现过的前缀和”以及它们出现的次数。
        #
        # We start with {0: 1} because before reading any number,
        # the prefix sum is 0 once.
        #
        # 初始化为 {0: 1}，因为在还没开始加任何数字之前，
        # 前缀和 0 已经出现过一次。
        seen = {0: 1}

        for num in nums:
            # Update current prefix sum.
            # 更新当前前缀和。
            cur_sum += num

            # We want:
            # current prefix sum - old prefix sum = k
            #
            # So:
            # old prefix sum = current prefix sum - k
            #
            # 我们想找：
            # 当前前缀和 - 旧前缀和 = k
            #
            # 所以：
            # 旧前缀和 = 当前前缀和 - k
            need = cur_sum - k

            # If this needed old prefix sum appeared before,
            # then each occurrence can form a valid subarray ending here.
            #
            # 如果 need 这个旧前缀和以前出现过，
            # 说明存在一个或多个以当前位置结尾、和为 k 的子数组。
            if need in seen:
                count += seen[need]

            # Record current prefix sum for future subarrays.
            # 把当前前缀和记录下来，供后面的元素使用。
            seen[cur_sum] = seen.get(cur_sum, 0) + 1

        return count


"""
My Learning Notes / 我的思路总结
--------------------------------

1. First idea: brute force with prefix sum
   一开始可以想到：先做 prefix 数组，再枚举所有 left 和 right。

   prefix[i] means the sum of nums[0] to nums[i - 1].
   prefix[i] 表示 nums[0] 到 nums[i - 1] 的和。

   So the sum of nums[left] to nums[right] is:
   所以 nums[left] 到 nums[right] 的区间和是：

       prefix[right + 1] - prefix[left]

   Brute force version:

       prefix = [0] * (len(nums) + 1)

       for i in range(len(nums)):
           prefix[i + 1] = prefix[i] + nums[i]

       count = 0

       for left in range(len(nums)):
           for right in range(left, len(nums)):
               current_sum = prefix[right + 1] - prefix[left]

               if current_sum == k:
                   count += 1

       return count

   This is correct, but it is O(n^2), so it may be too slow.
   这个方法是对的，但是时间复杂度是 O(n^2)，所以可能会超时。


2. Key optimization idea
   核心优化思路：

   In the brute force formula:
   在暴力法公式里：

       prefix[right + 1] - prefix[left] == k

   Move terms:
   移项：

       prefix[left] == prefix[right + 1] - k

   This means when we know the current prefix sum,
   we only need to check whether the needed old prefix sum appeared before.

   这说明：当我们知道当前前缀和时，
   只需要检查以前是否出现过“当前前缀和 - k”。


3. Why use a hashmap?
   为什么用哈希表？

   We use seen to record how many times each old prefix sum appeared.
   我们用 seen 记录每个旧前缀和出现过几次。

   If need appears 3 times before, then there are 3 different subarrays
   ending at the current position whose sum is k.

   如果 need 以前出现过 3 次，
   那就说明有 3 个不同的子数组以当前位置结尾，且和为 k。


4. Important code pattern
   重要代码模板：

       cur_sum += num
       need = cur_sum - k

       if need in seen:
           count += seen[need]

       seen[cur_sum] = seen.get(cur_sum, 0) + 1

   Meaning:
   含义：

   First check whether the needed old prefix sum exists.
   Then record the current prefix sum.

   先查以前有没有需要的旧前缀和，
   再把当前前缀和记录下来。


5. Time complexity
   时间复杂度：

   Brute force prefix sum: O(n^2)
   Optimized prefix sum + hashmap: O(n)

   暴力前缀和：O(n^2)
   前缀和 + 哈希表优化：O(n)
"""


if __name__ == "__main__":
    solution = Solution()

    nums = [1, 1, 1]
    k = 2

    print(solution.subarraySum(nums, k))  # Expected output: 2

    nums = [1, 2, 3]
    k = 3

    print(solution.subarraySum(nums, k))  # Expected output: 2
