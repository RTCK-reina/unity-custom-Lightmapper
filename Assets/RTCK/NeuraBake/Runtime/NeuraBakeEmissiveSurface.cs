using UnityEngine;

namespace RTCK.NeuraBake.Runtime
{
    [AddComponentMenu("RTCK/NeuraBake/Emissive Surface")]
    public class NeuraBakeEmissiveSurface : MonoBehaviour
    {
        [Tooltip("発光色")]
        public Color emissiveColor = Color.white;

        [Tooltip("発光強度")]
        [Min(0f)]
        public float intensity = 1.0f;

        [Tooltip("この面からの発光をライトマップベイクに含めるか")]
        public bool bakeEmissive = true;

        // 将来的に: テクスチャによる発光制御、両面発光オプションなど
    }
}