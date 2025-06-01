using System; // �� IProgress<T> �̂��߂ɒǉ� (�܂��͊m�F)
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace RTCK.NeuraBake.Runtime
{
    public interface ILightmapRenderer
    {
        Task<Texture2D> RenderAsync(CancellationToken token, IProgress<(float percentage, string message)> progress);
    }
}