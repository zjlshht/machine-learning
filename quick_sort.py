def quick_sort(l, r,arr):
    if l>=r:return
    i, j = l, r
    while i < j:
        while i < j and arr[j] >= arr[l]: j -= 1
        while i < j and arr[i] <= arr[l]: i += 1
        arr[i], arr[j] = arr[j], arr[i]
    arr[l], arr[i] = arr[i], arr[l]
    quick_sort(l, i - 1,arr) 
    quick_sort(i + 1, r,arr)
   

