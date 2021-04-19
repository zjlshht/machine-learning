def quick_sort(left, r, arr):
    if left >= r:
        return
    i, j = left, r
    while i < j:
        while i < j and arr[j] >= arr[left]:
            j -= 1
        while i < j and arr[i] <= arr[left]:
            i += 1
        arr[i], arr[j] = arr[j], arr[i]
    arr[left], arr[i] = arr[i], arr[left]
    quick_sort(left, i-1, arr)
    quick_sort(i+1, r, arr)
