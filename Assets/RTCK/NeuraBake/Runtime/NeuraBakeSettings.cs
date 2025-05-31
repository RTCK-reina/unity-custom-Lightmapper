using System;
using UnityEngine;

namespace RTCK.NeuraBake.Runtime
{
    [Serializable]
    public class NeuraBakeSettings
    {
        public static readonly float DefaultResolution = 40f;
        public static readonly int DefaultSampleCount = 64;
        public static readonly int DefaultBounceCount = 2;
        public static readonly bool DefaultUseAO = true;
        public static readonly bool DefaultDirectional = true;
        public static readonly bool DefaultUseDenoiser = false;
        public static readonly int DefaultAtlasSize = 1024;
        public static readonly int DefaultAoSampleCount = 16;
        public static readonly int DefaultShadowSamples = 4;
        public static readonly float DefaultEmissiveBoost = 1.0f;
        public static readonly float DefaultSkyIntensity = 1.0f;

        [Tooltip("ライトマップの解像度 (推奨: 20-60)")]
        [Range(1f, 100f)]
        public float resolution = DefaultResolution;

        [Tooltip("サンプル数 (多いほど品質が向上しますが、処理時間が増加します)")]
        [Range(1, 1024)]
        public int sampleCount = DefaultSampleCount;

        [Tooltip("ライトのバウンス数")]
        [Range(1, 10)]
        public int bounceCount = DefaultBounceCount;

        [Tooltip("環境光遮蔽を使用するかどうか")]
        public bool useAmbientOcclusion = DefaultUseAO;

        [Tooltip("指向性ライトマップを使用するかどうか")]
        public bool directional = DefaultDirectional;

        [Tooltip("デノイザーを使用してノイズを軽減")]
        public bool useDenoiser = DefaultUseDenoiser;

        [Tooltip("ライトマップアトラスのサイズ")]
        public int atlasSize = DefaultAtlasSize;

        [Tooltip("環境遮蔽光のサンプル数")]
        [Range(0, 64)]
        public int aoSampleCount = DefaultAoSampleCount;

        [Tooltip("影のサンプル数 (ソフトシャドウ用)")]
        [Range(1, 16)]
        public int shadowSamples = DefaultShadowSamples;

        [Tooltip("エミッシブサーフェスの強度倍率")]
        [Range(0.1f, 10.0f)]
        public float emissiveBoost = DefaultEmissiveBoost;

        [Tooltip("環境光の強度倍率")]
        [Range(0.0f, 5.0f)]
        public float skyIntensity = DefaultSkyIntensity;

        public NeuraBakeSettings() { }

        public void Reset()
        {
            resolution = DefaultResolution;
            sampleCount = DefaultSampleCount;
            bounceCount = DefaultBounceCount;
            useAmbientOcclusion = DefaultUseAO;
            directional = DefaultDirectional;
            useDenoiser = DefaultUseDenoiser;
            atlasSize = DefaultAtlasSize;
            aoSampleCount = DefaultAoSampleCount;
            shadowSamples = DefaultShadowSamples;
            emissiveBoost = DefaultEmissiveBoost;
            skyIntensity = DefaultSkyIntensity;
        }

        public (bool, string) Validate()
        {
            if (resolution < 1f || resolution > 100f)
                return (false, "解像度は1～100の範囲で指定してください。");
            if (sampleCount < 1 || sampleCount > 1024)
                return (false, "サンプル数は1～1024の範囲で指定してください。");
            if (bounceCount < 1 || bounceCount > 10)
                return (false, "バウンス数は1～10の範囲で指定してください。");
            if (aoSampleCount < 0 || aoSampleCount > 64)
                return (false, "AO サンプル数は0～64の範囲で指定してください。");
            if (shadowSamples < 1 || shadowSamples > 16)
                return (false, "影のサンプル数は1～16の範囲で指定してください。");
            if (emissiveBoost < 0.1f || emissiveBoost > 10.0f)
                return (false, "エミッシブ強度は0.1～10.0の範囲で指定してください。");
            if (skyIntensity < 0.0f || skyIntensity > 5.0f)
                return (false, "環境光強度は0.0～5.0の範囲で指定してください。");

            return (true, "");
        }
    }
}
