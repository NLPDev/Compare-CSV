def array1314(nums):

    result=0

    if len(nums)<4:
        return 0
    for i in range(len(nums)-3):
        if nums[i] == 1 and nums[i+1] == 3 and nums[i+2] == 1 and nums[i+3] == 4:
            result = result + 1

    return result


nums=[1, 2, 1, 3, 1, 4, 1, 4, 1, 3, 1, 4]

print(array1314(nums))