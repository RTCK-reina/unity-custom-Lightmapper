using System;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace RTCK.NeuraBake.Runtime
{
    public class CPU_Spectral_Renderer : ILightmapRenderer
    {
        private readonly NeuraBakeSettings settings;

        public CPU_Spectral_Renderer(NeuraBakeSettings settings)
        {
            this.settings = settings;
        }

        public async Task<Texture2D> RenderAsync(CancellationToken token, IProgress<(float percentage, string message)> progress)
        {
            progress?.Report((0.1f, "スペクトルレンダリング実証中..."));
            await Task.Delay(1000, token);
            if (token.IsCancellationRequested) return null;
            if (settings == null)
            {
                Debug.LogError("CPU_Spectral_Renderer: NeuraBakeSettings is null!");
                return new Texture2D(256, 256);
            }
            return new Texture2D(settings.atlasSize, settings.atlasSize);
        }
    }
}
