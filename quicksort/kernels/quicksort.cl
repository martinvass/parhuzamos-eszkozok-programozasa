__kernel void quicksort_segment(__global int* data, int left, int right)
{
    int stack[1024];
    int top = -1;

    stack[++top] = left;
    stack[++top] = right;

    while (top >= 0)
    {
        right = stack[top--];
        left = stack[top--];

        int i = left;
        int j = right;
        int pivot = data[(left + right) / 2];

        while (i <= j)
        {
            while (data[i] < pivot) i++;
            while (data[j] > pivot) j--;

            if (i <= j)
            {
                int tmp = data[i];
                data[i] = data[j];
                data[j] = tmp;
                i++;
                j--;
            }
        }

        if (left < j)
        {
            stack[++top] = left;
            stack[++top] = j;
        }

        if (i < right)
        {
            stack[++top] = i;
            stack[++top] = right;
        }
    }
}