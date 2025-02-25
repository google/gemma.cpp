using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
namespace GemmaCpp
{
    public class GemmaException : Exception
    {
        public GemmaException(string message) : base(message) { }
    }

    public class Gemma : IDisposable
    {
        private IntPtr _context;
        private bool _disposed;

        // Optional: Allow setting DLL path
        public static string DllPath { get; set; } = "gemma.dll";

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern IntPtr LoadLibrary(string lpFileName);

        static Gemma()
        {
            // Load DLL from specified path
            if (LoadLibrary(DllPath) == IntPtr.Zero)
            {
                throw new DllNotFoundException($"Failed to load {DllPath}. Error: {Marshal.GetLastWin32Error()}");
            }
        }

        [DllImport("gemma", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr GemmaCreate(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string tokenizerPath,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string modelType,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string weightsPath,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string weightType,
            int maxLength);

        [DllImport("gemma", CallingConvention = CallingConvention.Cdecl)]
        private static extern void GemmaDestroy(IntPtr context);

        // Delegate type for token callbacks
        public delegate bool TokenCallback(string token);

        // Keep delegate alive for duration of calls
        private GCHandle _callbackHandle;

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate bool GemmaTokenCallback(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string text,
            IntPtr userData);

        [DllImport("gemma", CallingConvention = CallingConvention.Cdecl)]
        private static extern int GemmaGenerate(
            IntPtr context,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string prompt,
            [MarshalAs(UnmanagedType.LPUTF8Str)] StringBuilder output,
            int maxLength,
            GemmaTokenCallback callback,
            IntPtr userData);

        [DllImport("gemma", CallingConvention = CallingConvention.Cdecl)]
        private static extern int GemmaCountTokens(
            IntPtr context,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string text);

        // Native callback delegate type
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void GemmaLogCallback(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string message,
            IntPtr userData);

        [DllImport("gemma", CallingConvention = CallingConvention.Cdecl)]
        private static extern void GemmaSetLogCallback(
            IntPtr context,
            GemmaLogCallback callback,
            IntPtr userData);

        private GCHandle _logCallbackHandle;

        public Gemma(string tokenizerPath, string modelType, string weightsPath, string weightType, int maxLength = 8192)
        {
            _context = GemmaCreate(tokenizerPath, modelType, weightsPath, weightType, maxLength);
            if (_context == IntPtr.Zero)
            {
                throw new GemmaException("Failed to create Gemma context");
            }

            // optionally: set up logging
            /*
            GemmaLogCallback logCallback = (message, _) =>
            {
#if UNITY_ENGINE
                Debug.Log($"Gemma: {message}");
#else
                Debug.WriteLine($"Gemma: {message}");
#endif
            };
            _logCallbackHandle = GCHandle.Alloc(logCallback);
            GemmaSetLogCallback(_context, logCallback, IntPtr.Zero);
            */
        }

        public int CountTokens(string prompt)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(Gemma));

            if (_context == IntPtr.Zero)
                throw new GemmaException("Gemma context is invalid");
            int count = GemmaCountTokens(_context, prompt);
            return count;
        }

        public string Generate(string prompt, int maxLength = 4096)
        {
            return Generate(prompt, null, maxLength);
        }

        public string Generate(string prompt, TokenCallback callback, int maxLength = 4096)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(Gemma));

            if (_context == IntPtr.Zero)
                throw new GemmaException("Gemma context is invalid");

            var output = new StringBuilder(maxLength);
            GemmaTokenCallback nativeCallback = null;

            if (callback != null)
            {
                nativeCallback = (text, _) => callback(text);
                _callbackHandle = GCHandle.Alloc(nativeCallback);
            }

            try
            {
                int length = GemmaGenerate(_context, prompt, output, maxLength,
                    nativeCallback, IntPtr.Zero);

                if (length < 0)
                    throw new GemmaException("Generation failed");

                return output.ToString();
            }
            finally
            {
                if (_callbackHandle.IsAllocated)
                    _callbackHandle.Free();
            }
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                if (_context != IntPtr.Zero)
                {
                    GemmaDestroy(_context);
                    _context = IntPtr.Zero;
                }
                if (_logCallbackHandle.IsAllocated)
                    _logCallbackHandle.Free();
                _disposed = true;
            }
        }

        ~Gemma()
        {
            Dispose();
        }
    }
}