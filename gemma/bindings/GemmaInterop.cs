// Copyright 2025 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
            int maxGeneratedTokens);

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
            [Out] byte[] output,
            int maxOutputChars,
            GemmaTokenCallback callback,
            IntPtr userData);

        [DllImport("gemma", CallingConvention = CallingConvention.Cdecl)]
        private static extern int GemmaGenerateMultimodal(
            IntPtr context,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string prompt,
            IntPtr image_data, // Renamed param to match C API
            int image_width,   // Added dimension
            int image_height,  // Added dimension
            [MarshalAs(UnmanagedType.LPUTF8Str)] StringBuilder output, // Output should be StringBuilder for multimodal
            int maxOutputChars,
            GemmaTokenCallback callback,
            IntPtr userData);

        [DllImport("gemma", CallingConvention = CallingConvention.Cdecl)]
        private static extern int GemmaCountTokens(
            IntPtr context,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string text);

        // Configuration function imports
        [DllImport("gemma", CallingConvention = CallingConvention.Cdecl)]
        private static extern void GemmaSetMaxGeneratedTokens(IntPtr context, int value);

        [DllImport("gemma", CallingConvention = CallingConvention.Cdecl)]
        private static extern void GemmaSetMultiturn(IntPtr context, int value);

        [DllImport("gemma", CallingConvention = CallingConvention.Cdecl)]
        private static extern void GemmaSetTemperature(IntPtr context, float value);

        [DllImport("gemma", CallingConvention = CallingConvention.Cdecl)]
        private static extern void GemmaSetTopK(IntPtr context, int value);

        [DllImport("gemma", CallingConvention = CallingConvention.Cdecl)]
        private static extern void GemmaSetDeterministic(IntPtr context, int value);

        [DllImport("gemma", CallingConvention = CallingConvention.Cdecl)]
        private static extern void GemmaSetPrefillTbatchSize(IntPtr context, int value);

        [DllImport("gemma", CallingConvention = CallingConvention.Cdecl, EntryPoint = "GemmaResetConversation")]
        private static extern void GemmaResetConversation(IntPtr context);

        // Conversation management function imports
        [DllImport("gemma", CallingConvention = CallingConvention.Cdecl, EntryPoint = "GemmaCreateConversation")]
        private static extern int GemmaCreateConversation(
            IntPtr context,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string conversationName);

        [DllImport("gemma", CallingConvention = CallingConvention.Cdecl, EntryPoint = "GemmaSwitchConversation")]
        private static extern int GemmaSwitchConversation(
            IntPtr context,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string conversationName);

        [DllImport("gemma", CallingConvention = CallingConvention.Cdecl, EntryPoint = "GemmaDeleteConversation")]
        private static extern int GemmaDeleteConversation(
            IntPtr context,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string conversationName);

        [DllImport("gemma", CallingConvention = CallingConvention.Cdecl, EntryPoint = "GemmaHasConversation")]
        private static extern int GemmaHasConversation(
            IntPtr context,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string conversationName);

        [DllImport("gemma", CallingConvention = CallingConvention.Cdecl, EntryPoint = "GemmaGetCurrentConversation")]
        [return: MarshalAs(UnmanagedType.LPUTF8Str)] // Marshal the const char* return value as a string
        private static extern string GemmaGetCurrentConversation(IntPtr context);

        [DllImport("gemma", CallingConvention = CallingConvention.Cdecl, EntryPoint = "GemmaSaveConversation")]
        private static extern void GemmaSaveConversation(IntPtr context);

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
        private bool _loggingEnabled = false;

        public Gemma(string tokenizerPath, string weightsPath, int maxGeneratedTokens = 8192)
        {
            _context = GemmaCreate(tokenizerPath, weightsPath, maxGeneratedTokens);
            if (_context == IntPtr.Zero)
            {
                throw new GemmaException("Failed to create Gemma context");
            }
        }

        // Enable debug logging
        public void EnableLogging(bool enable = true)
        {
            if (enable && !_loggingEnabled)
            {
                GemmaLogCallback logCallback = (message, _) =>
                {
                    Debug.WriteLine($"Gemma: {message}");
                };
                _logCallbackHandle = GCHandle.Alloc(logCallback);
                GemmaSetLogCallback(_context, logCallback, IntPtr.Zero);
                _loggingEnabled = true;
            }
            else if (!enable && _loggingEnabled)
            {
                if (_logCallbackHandle.IsAllocated)
                    _logCallbackHandle.Free();
                GemmaSetLogCallback(_context, null, IntPtr.Zero);
                _loggingEnabled = false;
            }
        }

        // Configuration methods
        public void SetMultiturn(bool enable)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(Gemma));

            if (_context == IntPtr.Zero)
                throw new GemmaException("Gemma context is invalid");

            GemmaSetMultiturn(_context, enable ? 1 : 0);
            Debug.WriteLine($"Gemma: Set multiturn to {(enable ? "enabled" : "disabled")}");
        }

        public void SetTemperature(float temperature)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(Gemma));

            if (_context == IntPtr.Zero)
                throw new GemmaException("Gemma context is invalid");

            GemmaSetTemperature(_context, temperature);
            Debug.WriteLine($"Gemma: Set temperature to {temperature}");
        }

        public void SetTopK(int topK)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(Gemma));

            if (_context == IntPtr.Zero)
                throw new GemmaException("Gemma context is invalid");

            GemmaSetTopK(_context, topK);
            Debug.WriteLine($"Gemma: Set topK to {topK}");
        }

        public void SetDeterministic(bool deterministic)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(Gemma));

            if (_context == IntPtr.Zero)
                throw new GemmaException("Gemma context is invalid");

            GemmaSetDeterministic(_context, deterministic ? 1 : 0);
            Debug.WriteLine($"Gemma: Set deterministic to {(deterministic ? "true" : "false")}");
        }

        // Renamed public method
        public void ResetConversation()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(Gemma));

            if (_context == IntPtr.Zero)
                throw new GemmaException("Gemma context is invalid");

            GemmaResetConversation(_context); // Call P/Invoke method
            Debug.WriteLine("Gemma: Reset active conversation");
        }

        // Conversation management methods
        public bool CreateConversation(string conversationName)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(Gemma));

            if (_context == IntPtr.Zero)
                throw new GemmaException("Gemma context is invalid");

            bool result = GemmaCreateConversation(_context, conversationName) != 0; // Call P/Invoke method
            Debug.WriteLine($"Gemma: Create conversation '{conversationName}' - {(result ? "succeeded" : "failed")}");
            return result;
        }

        public bool SwitchConversation(string conversationName)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(Gemma));

            if (_context == IntPtr.Zero)
                throw new GemmaException("Gemma context is invalid");

            bool result = GemmaSwitchConversation(_context, conversationName) != 0; // Call P/Invoke method
            Debug.WriteLine($"Gemma: Switch to conversation '{conversationName}' - {(result ? "succeeded" : "failed")}");
            return result;
        }

        public bool DeleteConversation(string conversationName)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(Gemma));

            if (_context == IntPtr.Zero)
                throw new GemmaException("Gemma context is invalid");

            bool result = GemmaDeleteConversation(_context, conversationName) != 0; // Call P/Invoke method
            Debug.WriteLine($"Gemma: Delete conversation '{conversationName}' - {(result ? "succeeded" : "failed")}");
            return result;
        }

        public bool HasConversation(string conversationName)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(Gemma));

            if (_context == IntPtr.Zero)
                throw new GemmaException("Gemma context is invalid");

            bool result = GemmaHasConversation(_context, conversationName) != 0; // Call P/Invoke method
            Debug.WriteLine($"Gemma: Has conversation '{conversationName}' - {result}");
            return result;
        }

        public string GetCurrentConversation()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(Gemma));

            if (_context == IntPtr.Zero)
                throw new GemmaException("Gemma context is invalid");

            string currentConversation = GemmaGetCurrentConversation(_context); // Call P/Invoke method
            Debug.WriteLine($"Gemma: Current conversation is '{currentConversation}'");
            return currentConversation;
        }

        public void SaveConversation()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(Gemma));

            if (_context == IntPtr.Zero)
                throw new GemmaException("Gemma context is invalid");

            GemmaSaveConversation(_context);
            Debug.WriteLine($"Gemma: Saved current conversation ('{GetCurrentConversation()}') to prewarmed cache.");
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

        public string Generate(string prompt, int maxOutputChars = 4096)
        {
            return Generate(prompt, null, maxOutputChars);
        }

        public string Generate(string prompt, TokenCallback callback, int maxOutputChars = 4096)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(Gemma));

            if (_context == IntPtr.Zero)
                throw new GemmaException("Gemma context is invalid");

            var outputBuffer = new byte[maxOutputChars * 4];  // Allow for worst case UTF-8 size
            GemmaTokenCallback nativeCallback = null;

            // Track token count for debugging
            int tokenCount = 0;

            if (callback != null)
            {
                nativeCallback = (text, _) =>
                {
                    tokenCount++;
                    // Log token for debugging
                    Debug.WriteLine($"Token {tokenCount}: '{text}'");

                    // Pass token to user callback
                    return callback(text);
                };
                _callbackHandle = GCHandle.Alloc(nativeCallback);
            }

            try
            {
                int length = GemmaGenerate(_context, prompt, outputBuffer, maxOutputChars,
                    nativeCallback, IntPtr.Zero);

                if (length < 0)
                    throw new GemmaException("Generation failed");

                Debug.WriteLine($"Generation complete: {tokenCount} tokens processed, result length: {length}");

                // Convert the byte buffer to a string using UTF-8 encoding
                string result = Encoding.UTF8.GetString(outputBuffer, 0, length);
                return result;
            }
            finally
            {
                if (_callbackHandle.IsAllocated)
                    _callbackHandle.Free();
            }
        }

        public string GenerateMultimodal(string prompt, float[] imageData, int imageWidth, int imageHeight, int maxOutputChars = 4096)
        {
            // Pass width and height to the overloaded method
            return GenerateMultimodal(prompt, imageData, imageWidth, imageHeight, null, maxOutputChars);
        }

        public string GenerateMultimodal(string prompt, float[] imageData, int imageWidth, int imageHeight, TokenCallback callback, int maxOutputChars = 4096)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(Gemma));

            if (_context == IntPtr.Zero)
                throw new GemmaException("Gemma context is invalid");

            if (imageData == null || imageData.Length == 0)
                throw new ArgumentException("Image data cannot be null or empty", nameof(imageData));

            if (imageWidth <= 0 || imageHeight <= 0)
                throw new ArgumentException("Image dimensions must be positive");

            if (imageData.Length < imageWidth * imageHeight * 3)
                throw new ArgumentException("Image data array is too small for the specified dimensions");

            var output = new StringBuilder(maxOutputChars);
            GemmaTokenCallback nativeCallback = null;

            if (callback != null)
            {
                nativeCallback = (text, _) => callback(text);
                _callbackHandle = GCHandle.Alloc(nativeCallback);
            }

            // Pin the image data so it doesn't move during the native call
            GCHandle imageHandle = GCHandle.Alloc(imageData, GCHandleType.Pinned);

            try
            {
                IntPtr imagePtr = imageHandle.AddrOfPinnedObject();

                // Pass image dimensions to the native call
                int length = GemmaGenerateMultimodal(_context, prompt, imagePtr, imageWidth, imageHeight, output, maxOutputChars,
                    nativeCallback, IntPtr.Zero);

                if (length < 0)
                    throw new GemmaException("Multimodal generation failed");

                return output.ToString();
            }
            finally
            {
                imageHandle.Free();

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
