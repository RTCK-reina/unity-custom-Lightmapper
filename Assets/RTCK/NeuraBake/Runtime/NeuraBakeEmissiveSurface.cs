using UnityEngine;

namespace RTCK.NeuraBake.Runtime
{
    [AddComponentMenu("RTCK/NeuraBake/Emissive Surface")]
    public class NeuraBakeEmissiveSurface : MonoBehaviour
    {
        [Tooltip("�����F")]
        public Color emissiveColor = Color.white;

        [Tooltip("�������x")]
        [Min(0f)]
        public float intensity = 1.0f;

        [Tooltip("���̖ʂ���̔��������C�g�}�b�v�x�C�N�Ɋ܂߂邩")]
        public bool bakeEmissive = true;

        // �����I��: �e�N�X�`���ɂ�锭������A���ʔ����I�v�V�����Ȃ�
    }
}