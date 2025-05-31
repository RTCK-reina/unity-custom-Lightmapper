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

        [Tooltip("���C�g�}�b�v�̉𑜓x (����: 20-60)")]
        [Range(1f, 100f)]
        public float resolution = DefaultResolution;

        [Tooltip("�T���v���� (�����قǕi�������サ�܂����A�������Ԃ��������܂�)")]
        [Range(1, 1024)]
        public int sampleCount = DefaultSampleCount;

        [Tooltip("���C�g�̃o�E���X��")]
        [Range(1, 10)]
        public int bounceCount = DefaultBounceCount;

        [Tooltip("�����Օ����g�p���邩�ǂ���")]
        public bool useAmbientOcclusion = DefaultUseAO;

        [Tooltip("�w�������C�g�}�b�v���g�p���邩�ǂ���")]
        public bool directional = DefaultDirectional;

        [Tooltip("�f�m�C�U�[���g�p���ăm�C�Y���y��")]
        public bool useDenoiser = DefaultUseDenoiser;

        [Tooltip("���C�g�}�b�v�A�g���X�̃T�C�Y")]
        public int atlasSize = DefaultAtlasSize;

        [Tooltip("���Օ����̃T���v����")]
        [Range(0, 64)]
        public int aoSampleCount = DefaultAoSampleCount;

        [Tooltip("�e�̃T���v���� (�\�t�g�V���h�E�p)")]
        [Range(1, 16)]
        public int shadowSamples = DefaultShadowSamples;

        [Tooltip("�G�~�b�V�u�T�[�t�F�X�̋��x�{��")]
        [Range(0.1f, 10.0f)]
        public float emissiveBoost = DefaultEmissiveBoost;

        [Tooltip("�����̋��x�{��")]
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
                return (false, "�𑜓x��1�`100�͈̔͂Ŏw�肵�Ă��������B");
            if (sampleCount < 1 || sampleCount > 1024)
                return (false, "�T���v������1�`1024�͈̔͂Ŏw�肵�Ă��������B");
            if (bounceCount < 1 || bounceCount > 10)
                return (false, "�o�E���X����1�`10�͈̔͂Ŏw�肵�Ă��������B");
            if (aoSampleCount < 0 || aoSampleCount > 64)
                return (false, "AO �T���v������0�`64�͈̔͂Ŏw�肵�Ă��������B");
            if (shadowSamples < 1 || shadowSamples > 16)
                return (false, "�e�̃T���v������1�`16�͈̔͂Ŏw�肵�Ă��������B");
            if (emissiveBoost < 0.1f || emissiveBoost > 10.0f)
                return (false, "�G�~�b�V�u���x��0.1�`10.0�͈̔͂Ŏw�肵�Ă��������B");
            if (skyIntensity < 0.0f || skyIntensity > 5.0f)
                return (false, "�������x��0.0�`5.0�͈̔͂Ŏw�肵�Ă��������B");

            return (true, "");
        }
    }
}
