#undef  THRUST_DEBUG
#define THRUST_DEBUG 1

#include <thrust/detail/backend/cuda/reduce_intervals.h>
#include <thrust/detail/backend/cuda/reduce_by_key.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <iostream>

#include <ToString.h>

using namespace thrust::detail::backend::cuda;

using std::cout;
using std::endl;

template <typename InputIterator,
          typename OutputIterator,
          typename BinaryFunction,
          typename Decomposition>
struct local_commutative_reduce_intervals_closure
{
  InputIterator  input;
  OutputIterator output;
  BinaryFunction binary_op;
  Decomposition  decomposition;
  unsigned int shared_array_size;

  local_commutative_reduce_intervals_closure(InputIterator input, OutputIterator output, BinaryFunction binary_op, Decomposition decomposition, unsigned int shared_array_size)
    : input(input), output(output), binary_op(binary_op), decomposition(decomposition), shared_array_size(shared_array_size) {
      cout << "create 'local_commutative_reduce_intervals_closure'" << endl;
      typedef typename Decomposition::index_type index_type;
      thrust::detail::backend::index_range<index_type> firstRange = decomposition[0];
      cout << "first range: " << firstRange.begin() << " -> " << firstRange.end() << ", size: " << firstRange.size() << endl;

      output[0] = 10;
      cout << "input[0] " << input[0] << 
          ", input[1] " << input[1] << 
          ", input[2] " << input[2] << 
          ", input[3] " << input[3] << 
          ", input[4] " << input[4] << 
          ", input[5] " << input[5] << 
          ", input[6] " << input[6] << endl;
      cout << "output[0]" << output[0] << endl;
  }

  __device__ 
  void operator()(void) {
      // reduce_n uses built-in variables
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC
      typedef typename Decomposition::index_type index_type;
      // this block processes results in [range.begin(), range.end())
      thrust::detail::backend::index_range<index_type> range = decomposition[blockIdx.x];
      //thrust::detail::backend::index_range<index_type> range(0, 7);

      typedef typename thrust::iterator_value<OutputIterator>::type OutputType;
      thrust::detail::backend::cuda::extern_shared_ptr<OutputType>  shared_array;

      if (threadIdx.x == 0) {
          output += blockIdx.x;
          dereference(output) = dereference(input);
      }

      index_type i = range.begin() + threadIdx.x;
      input += i;
      
      // compute reduction with the first shared_array_size threads
      if (threadIdx.x < thrust::min<index_type>(shared_array_size,range.size())) {
          OutputType sum = dereference(input);

          i     += shared_array_size;
          input += shared_array_size;

          while (i < range.end()) {
              OutputType val = dereference(input);
              
              sum = binary_op(sum, val);
              
              i      += shared_array_size;
              input  += shared_array_size;
          }
          shared_array[threadIdx.x] = sum;
          
          shared_array[threadIdx.x] = dereference(input);
      }

      __syncthreads(); 
      
      thrust::detail::backend::cuda::block::reduce_n(shared_array, thrust::min<index_type>(range.size(), shared_array_size), binary_op);
      if (threadIdx.x == 0) {
          output += blockIdx.x;
          dereference(output) = shared_array[0];
      }
#endif // THRUST_DEVICE_COMPILER_NVCC
  }
};

template <typename InputIterator,
          typename OutputIterator,
          typename BinaryFunction,
          typename Decomposition>
void local_reduce_intervals(InputIterator input,
                            OutputIterator output,
                            BinaryFunction binary_op,
                            Decomposition decomp) {
  // we're attempting to launch a kernel, assert we're compiling with nvcc
  // ========================================================================
  // X Note to the user: If you've found this line due to a compiler error, X
  // X you need to compile your code using nvcc, rather than g++ or cl.exe  X
  // ========================================================================
  // THRUST_STATIC_ASSERT( (depend_on_instantiation<InputIterator, THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC>::value) );

  if (decomp.size() == 0) return;
  
  // TODO if (decomp.size() > deviceProperties.maxGridSize[0]) throw cuda exception (or handle general case)
  
  typedef local_commutative_reduce_intervals_closure<InputIterator,OutputIterator,BinaryFunction,Decomposition> Closure;
  thrust::detail::backend::cuda::detail::launch_calculator<Closure> calculator;
  thrust::tuple<size_t,size_t,size_t> config = calculator.with_variable_block_size_available_smem();

  size_t block_size = thrust::get<1>(config);
  size_t max_memory = thrust::get<2>(config);

  // determine shared array size
  typedef typename thrust::iterator_value<OutputIterator>::type OutputType;
  size_t shared_array_size  = thrust::min(max_memory / sizeof(OutputType), block_size);
  size_t shared_array_bytes = sizeof(OutputType) * shared_array_size;
  
  // TODO if (shared_array_size < 1) throw cuda exception "insufficient shared memory"

  Closure closure(input, output, binary_op, decomp, shared_array_size);

  cout << "Launching " << decomp.size() << " blocks of kernel with " << block_size << " threads and " << shared_array_bytes << "bytes of  shared memory per block " << endl;
  cout << "sizeof closure: " << sizeof(Closure) << endl;
  
  thrust::detail::backend::cuda::detail::launch_closure(closure, decomp.size(), block_size, shared_array_bytes);
}


template <typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2,
          typename BinaryPredicate,
          typename BinaryFunction>
void local_reduce_by_key(InputIterator1 keys_first, 
                         InputIterator1 keys_last,
                         InputIterator2 values_first,
                         OutputIterator1 keys_output,
                         OutputIterator2 values_output,
                         BinaryPredicate binary_pred,
                         BinaryFunction binary_op) {
    typedef          unsigned int                                              FlagType;
    typedef typename thrust::iterator_traits<InputIterator1>::difference_type  IndexType;
    //typedef unsigned int  IndexType;
    typedef typename thrust::iterator_traits<InputIterator1>::value_type       KeyType;
    
    cout << sizeof(typename thrust::iterator_traits<InputIterator1>::difference_type) << " VS " << sizeof(unsigned int) << endl;

    typedef thrust::detail::backend::uniform_decomposition<IndexType> Decomposition;
   
    // temporary arrays
    typedef thrust::detail::uninitialized_array<IndexType,thrust::detail::cuda_device_space_tag> IndexArray;

    // input size
    IndexType n = keys_last - keys_first;
    cout << "n: " << n << endl;
    if (n == 0) return;
 
    Decomposition decomp = thrust::detail::backend::cuda::default_decomposition<IndexType>(n);
    IndexArray interval_counts(decomp.size());

    // count number of tail flags per interval
    local_reduce_intervals(thrust::make_transform_iterator
                           (thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<IndexType>(0), keys_first, keys_first + 1)),
                            detail::tail_flag_functor<FlagType,IndexType,KeyType,BinaryPredicate>(n, binary_pred)), 
                           interval_counts.begin(), thrust::plus<IndexType>(), decomp);
    
    cout << interval_counts[0] << endl;
}

struct Assign10 {
    thrust::device_vector<long>::iterator output;
    
    Assign10(thrust::device_vector<long>::iterator output)
        : output(output) {}

    __device__    
    void operator()(const unsigned int threadId) const {
        dereference(output + threadId) = 10;
    }
};

int main(int argc, char *argv[]){
    /*
    thrust::device_vector<long> out(10);
    thrust::for_each(thrust::counting_iterator<unsigned int>(0), 
                     thrust::counting_iterator<unsigned int>(out.size()),
                     Assign10(out.begin()));
    cout << out << endl;
    */    

    const int N = 7;
    thrust::device_vector<int> A(N);
    A[0] = 1; A[1] = 3; A[2] = 3; A[3] = 3; A[4] = 2; A[5] = 2; A[6] = 1;
    thrust::device_vector<int> B(N);
    B[0] = 9; B[1] = 8; B[2] = 7; B[3] = 6; B[4] = 5; B[5] = 4; B[6] = 3;
    thrust::device_vector<int> C(N);
    thrust::device_vector<int> D(N);
    
    thrust::equal_to<int> binary_pred;
    thrust::plus<int> binary_op;
    local_reduce_by_key(A.begin(), A.end(), B.begin(), C.begin(), D.begin(), 
                        binary_pred, binary_op);
}
