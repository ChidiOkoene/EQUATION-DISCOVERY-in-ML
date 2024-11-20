# For Loop

def average_l(nums: list) -> float:
    for n in nums:
        n = float(n)
    
    L = len(nums)

    summation = sum([n for n in nums])

    ave = summation / L

    return ave

numbers = [1,5,4,2,5,8,4,6,8]
meanM = average_l(numbers)
print(meanM)
