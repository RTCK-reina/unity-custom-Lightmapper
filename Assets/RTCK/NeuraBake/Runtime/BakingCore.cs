using UnityEngine;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Threading;
using System.Linq;
using System;
using Random = UnityEngine.Random; // UnityEngine.Random �𖾎�

namespace RTCK.NeuraBake.Runtime
{
    public class BakingCore
    {
        private readonly NeuraBakeSettings settings;
        private readonly Light[] sceneLights;
        private readonly NeuraBakeEmissiveSurface[] emissiveSurfaces; // �����ʂ��L���b�V��
        private readonly Dictionary<Material, MaterialProperties> materialCache = new Dictionary<Material, MaterialProperties>();
        private readonly Dictionary<Mesh, MeshDataCache> meshDataCache = new Dictionary<Mesh, MeshDataCache>();
        private readonly object cacheLock = new object();

        private class MaterialProperties // �O�񂩂�ύX�Ȃ��̂��ߏȗ� (�������AGetSampledMetallicRoughness����Smoothness�̈������Ċm�F)
        {
            public Color BaseColor { get; }
            public float Metallic { get; }
            public float Roughness { get; }
            public Texture2D BaseColorMap { get; }
            public Texture2D MetallicGlossMap { get; }
            public Texture2D NormalMap { get; }
            public Vector4 MainTexST { get; }

            public MaterialProperties(Material material)
            {
                BaseColor = material.HasProperty("_BaseColor") ? material.GetColor("_BaseColor") :
                            (material.HasProperty("_Color") ? material.GetColor("_Color") : Color.white);
                Metallic = material.HasProperty("_Metallic") ? material.GetFloat("_Metallic") : 0f;
                float smoothness = material.HasProperty("_Glossiness") ? material.GetFloat("_Glossiness") :
                                  (material.HasProperty("_Smoothness") ? material.GetFloat("_Smoothness") : 0.5f);
                Roughness = 1f - smoothness;
                BaseColorMap = material.HasProperty("_MainTex") ? material.GetTexture("_MainTex") as Texture2D : null;
                MetallicGlossMap = material.HasProperty("_MetallicGlossMap") ? material.GetTexture("_MetallicGlossMap") as Texture2D : null;
                NormalMap = material.HasProperty("_BumpMap") ? material.GetTexture("_BumpMap") as Texture2D : null;
                MainTexST = material.HasProperty("_MainTex_ST") ? material.GetVector("_MainTex_ST") : new Vector4(1, 1, 0, 0);
            }

            public Color GetSampledBaseColor(Vector2 uv)
            {
                Vector2 tiledUv = new Vector2(uv.x * MainTexST.x + MainTexST.z, uv.y * MainTexST.y + MainTexST.w);
                Color texColor = BaseColorMap != null && BaseColorMap.isReadable ? BaseColorMap.GetPixelBilinear(tiledUv.x, tiledUv.y) : Color.white;
                // sRGB�e�N�X�`���̏ꍇ�A���j�A��Ԃɕϊ� (Unity�������ōs���ꍇ�����邪�A�����ł͖������Ȃ��B�ŏI�o�͂����j�A�Ȃ�OK)
                return BaseColor * texColor;
            }

            public (float sampledMetallic, float sampledRoughness) GetSampledMetallicRoughness(Vector2 uv)
            {
                Vector2 tiledUv = new Vector2(uv.x * MainTexST.x + MainTexST.z, uv.y * MainTexST.y + MainTexST.w);
                float finalMetallic = Metallic;
                float currentSmoothness = 1f - Roughness; // Roughness����Smoothness�ɖ߂�

                if (MetallicGlossMap != null && MetallicGlossMap.isReadable)
                {
                    Color metallicGlossSample = MetallicGlossMap.GetPixelBilinear(tiledUv.x, tiledUv.y);
                    finalMetallic *= metallicGlossSample.r; // Standard shader: Metallic is in R channel
                    currentSmoothness *= metallicGlossSample.a; // Standard shader: Smoothness is in A channel
                }
                return (Mathf.Clamp01(finalMetallic), 1f - Mathf.Clamp01(currentSmoothness)); // �ēxRoughness�ɕϊ�
            }
        }


        private class MeshDataCache // �O�񂩂�ύX�Ȃ��̂��ߏȗ�
        {
            public readonly Vector3[] Vertices;
            public readonly Vector3[] Normals;
            public readonly Vector2[] UVs;
            public readonly int[] Triangles;
            public MeshDataCache(Mesh mesh)
            {
                Vertices = mesh.vertices;
                Normals = mesh.normals;
                UVs = mesh.uv2 != null && mesh.uv2.Length == mesh.vertexCount ? mesh.uv2 : mesh.uv;
                Triangles = mesh.triangles;
            }
        }


        public BakingCore(NeuraBakeSettings bakeSettings)
        {
            settings = bakeSettings ?? throw new ArgumentNullException(nameof(bakeSettings));
            sceneLights = GameObject.FindObjectsOfType<Light>().Where(l => l.isActiveAndEnabled && l.intensity > 0).ToArray();
            // �����ʃR���|�[�l���g�����I�u�W�F�N�g�����W
            emissiveSurfaces = GameObject.FindObjectsOfType<NeuraBakeEmissiveSurface>()
                                         .Where(es => es.enabled && es.bakeEmissive && es.intensity > 0)
                                         .ToArray();
        }

        private MaterialProperties GetMaterialProperties(Material material) // �O�񂩂�ύX�Ȃ�
        {
            lock (cacheLock)
            {
                if (!materialCache.TryGetValue(material, out MaterialProperties props))
                {
                    props = new MaterialProperties(material);
                    materialCache[material] = props;
                }
                return props;
            }
        }

        private MeshDataCache GetMeshData(Mesh mesh) // �O�񂩂�ύX�Ȃ�
        {
            lock (cacheLock)
            {
                if (!meshDataCache.TryGetValue(mesh, out MeshDataCache data))
                {
                    data = new MeshDataCache(mesh);
                    meshDataCache[mesh] = data;
                }
                return data;
            }
        }

        // BakeLightmapAsync �Ƃ��̃w���p�[ (FindTriangleAndBarycentricCoords, InterpolateVector3/2) �͑O�񂩂�قڕύX�Ȃ��B
        // CalculatePixelColor �ɓn�� surfaceUV �̃W�b�^�����O�����𖾊m���B
        public async Task<Texture2D> BakeLightmapAsync(CancellationToken token, IProgress<(float percentage, string message)> progressReporter)
        {
            MeshRenderer[] renderers = GameObject.FindObjectsOfType<MeshRenderer>()
                                               .Where(r => r.enabled && r.gameObject.isStatic).ToArray();

            if (renderers.Length == 0)
            {
                progressReporter?.Report((1f, "�x�C�N�Ώۂ̐ÓIMeshRenderer��������܂���B"));
                return null;
            }

            MeshRenderer targetRenderer = renderers[0]; // �t�F�[�Y1�͍ŏ��̃����_���[�̂�
            Material targetMaterial = targetRenderer.sharedMaterial;
            MeshFilter meshFilter = targetRenderer.GetComponent<MeshFilter>();

            if (meshFilter == null || meshFilter.sharedMesh == null || targetMaterial == null)
            {
                progressReporter?.Report((1f, "�ΏۃI�u�W�F�N�g�Ƀ��b�V���܂��̓}�e���A��������܂���B"));
                return null;
            }

            Mesh mesh = meshFilter.sharedMesh;
            MeshDataCache meshData = GetMeshData(mesh);

            if (meshData.UVs == null || meshData.UVs.Length == 0)
            {
                progressReporter?.Report((1f, "�Ώۃ��b�V���Ƀ��C�g�}�b�v�p��UV������܂���B"));
                return null;
            }

            int textureWidth = settings.atlasSize;
            int textureHeight = settings.atlasSize;
            Texture2D lightmapTexture = new Texture2D(textureWidth, textureHeight, TextureFormat.RGBAHalf, false, true);
            lightmapTexture.name = $"{targetRenderer.gameObject.name}_Lightmap_Baked";
            lightmapTexture.wrapMode = TextureWrapMode.Clamp;
            lightmapTexture.filterMode = FilterMode.Bilinear;

            Color[] pixels = new Color[textureWidth * textureHeight];
            int totalPixelsToProcess = textureWidth * textureHeight;
            int processedPixels = 0;
            float accumulatedBentNormalY = 0f; // �x���g�m�[�}��Y�v�Z�p

            for (int y = 0; y < textureHeight; y++)
            {
                for (int x = 0; x < textureWidth; x++)
                {
                    token.ThrowIfCancellationRequested();
                    Vector2 texelCenterUv = new Vector2((x + 0.5f) / textureWidth, (y + 0.5f) / textureHeight);

                    if (FindTriangleAndBarycentricCoords(meshData, texelCenterUv, out int triangleIndex, out Vector3 barycentricCoords))
                    {
                        int vIdx0 = meshData.Triangles[triangleIndex * 3 + 0];
                        int vIdx1 = meshData.Triangles[triangleIndex * 3 + 1];
                        int vIdx2 = meshData.Triangles[triangleIndex * 3 + 2];

                        Vector3 localPos = InterpolateVector3(meshData.Vertices[vIdx0], meshData.Vertices[vIdx1], meshData.Vertices[vIdx2], barycentricCoords);
                        Vector3 localNormal = InterpolateVector3(meshData.Normals[vIdx0], meshData.Normals[vIdx1], meshData.Normals[vIdx2], barycentricCoords).normalized;

                        // ���C�g�}�b�v�p��UV���W���� (�}�e���A���T���v�����O�Ƃ͕ʂ̏ꍇ������)
                        Vector2 lightmapUV = InterpolateVector2(meshData.UVs[vIdx0], meshData.UVs[vIdx1], meshData.UVs[vIdx2], barycentricCoords);


                        Vector3 worldPos = targetRenderer.transform.TransformPoint(localPos);
                        Vector3 worldNormal = targetRenderer.transform.TransformDirection(localNormal).normalized;

                        Color accumulatedColor = Color.black;
                        accumulatedBentNormalY = 0f; // �s�N�Z�����ƂɃ��Z�b�g
                        int validAoSamplesForBentNormal = 0;

                        for (int s = 0; s < settings.sampleCount; s++) // �X�[�p�[�T���v�����O���[�v
                        {
                            Vector2 currentSampleUV = texelCenterUv;
                            if (settings.sampleCount > 1) // 1�ȏ�̎��̂݃W�b�^�����O
                            {
                                float offsetX = (Random.value - 0.5f) / textureWidth; // 1�s�N�Z�������Ń����_��
                                float offsetY = (Random.value - 0.5f) / textureHeight;
                                currentSampleUV = new Vector2(texelCenterUv.x + offsetX, texelCenterUv.y + offsetY);
                                // �W�b�^�����O����UV���O�p�`�̊O�ɏo��\���͒Ⴂ���A�����ɂ͍ă`�F�b�N���K�v
                            }

                            // CalculatePixelColor�� (Color, BentNormalYComponent) ��Ԃ��悤�ɕύX
                            (Color pixelSampleColor, float sampleBentNormalY, int unoccludedRays) colorInfo = CalculatePixelColor(worldPos, worldNormal, targetMaterial, lightmapUV, currentSampleUV /*�}�e���A���pUV*/);
                            accumulatedColor += colorInfo.pixelSampleColor;
                            if (settings.directional && colorInfo.unoccludedRays > 0) // unoccludedRays��AO�v�Z���ɃJ�E���g
                            {
                                accumulatedBentNormalY += colorInfo.sampleBentNormalY * colorInfo.unoccludedRays; // �d�ݕt�����ς̂���
                                validAoSamplesForBentNormal += colorInfo.unoccludedRays;
                            }
                        }
                        Color finalPixelColor = accumulatedColor / settings.sampleCount;

                        if (settings.directional)
                        {
                            if (validAoSamplesForBentNormal > 0)
                            {
                                finalPixelColor.a = Mathf.Clamp01(0.5f + (accumulatedBentNormalY / validAoSamplesForBentNormal) * 0.5f); // -1..1 to 0..1 range
                            }
                            else if (settings.useAmbientOcclusion && settings.aoSampleCount > 0) // AO�������S��
                            {
                                finalPixelColor.a = 0f; //���S�ɉ������i�n�ʂȂǁj�̋ߎ�
                            }
                            else // AO�Ȃ��A�܂��̓T���v��0
                            {
                                finalPixelColor.a = 0.5f; //���������Ȃ� (�j���[�g����)
                            }
                        }
                        else
                        {
                            finalPixelColor.a = 1.0f; // �A���t�@�g��Ȃ��ꍇ��1
                        }
                        pixels[y * textureWidth + x] = finalPixelColor;
                    }
                    else
                    {
                        pixels[y * textureWidth + x] = Color.clear;
                    }
                    processedPixels++;
                }
                if (y % 10 == 0 || y == textureHeight - 1)
                {
                    float percentage = (float)processedPixels / totalPixelsToProcess;
                    await Task.Yield();
                    progressReporter?.Report((percentage, $"�s�N�Z�������� ({y + 1}/{textureHeight}�s)"));
                }
            }

            lightmapTexture.SetPixels(pixels);
            lightmapTexture.Apply(true, false);
            progressReporter?.Report((1f, "���C�g�}�b�v��������"));
            return lightmapTexture;
        }

        // FindTriangleAndBarycentricCoords, CalculateBarycentricCoords, InterpolateVector3/2 �͑O�񂩂�ύX�Ȃ�
        private bool FindTriangleAndBarycentricCoords(MeshDataCache meshData, Vector2 uv, out int triangleStartVertexIndex, out Vector3 barycentricCoords)
        {
            triangleStartVertexIndex = 0;
            barycentricCoords = Vector3.zero;
            for (int i = 0; i < meshData.Triangles.Length / 3; i++)
            {
                Vector2 uv0 = meshData.UVs[meshData.Triangles[i * 3 + 0]];
                Vector2 uv1 = meshData.UVs[meshData.Triangles[i * 3 + 1]];
                Vector2 uv2 = meshData.UVs[meshData.Triangles[i * 3 + 2]];
                barycentricCoords = CalculateBarycentricCoords(uv, uv0, uv1, uv2);
                if (barycentricCoords.x >= -1e-4f && barycentricCoords.y >= -1e-4f && barycentricCoords.z >= -1e-4f &&
                    barycentricCoords.x <= 1.0001f && barycentricCoords.y <= 1.0001f && barycentricCoords.z <= 1.0001f)
                {
                    triangleStartVertexIndex = i;
                    return true;
                }
            }
            return false;
        }
        private Vector3 CalculateBarycentricCoords(Vector2 p, Vector2 a, Vector2 b, Vector2 c)
        {
            Vector2 v0 = b - a, v1 = c - a, v2 = p - a;
            float d00 = Vector2.Dot(v0, v0);
            float d01 = Vector2.Dot(v0, v1);
            float d11 = Vector2.Dot(v1, v1);
            float d20 = Vector2.Dot(v2, v0);
            float d21 = Vector2.Dot(v2, v1);
            float denom = d00 * d11 - d01 * d01;
            if (Mathf.Abs(denom) < 1e-6f) return new Vector3(-1, -1, -1);
            float v_ = (d11 * d20 - d01 * d21) / denom; // Renamed v to v_
            float w_ = (d00 * d21 - d01 * d20) / denom; // Renamed w to w_
            float u_ = 1.0f - v_ - w_;                 // Renamed u to u_
            return new Vector3(u_, v_, w_);
        }
        private Vector3 InterpolateVector3(Vector3 v0, Vector3 v1, Vector3 v2, Vector3 barycentric)
        { return v0 * barycentric.x + v1 * barycentric.y + v2 * barycentric.z; }
        private Vector2 InterpolateVector2(Vector2 v0, Vector2 v1, Vector2 v2, Vector3 barycentric)
        { return v0 * barycentric.x + v1 * barycentric.y + v2 * barycentric.z; }


        /// <summary>
        /// �s�N�Z���̐F�ƃx���g�m�[�}��Y�������v�Z���܂��B
        /// </summary>
        /// <returns> (�v�Z���ꂽ�F, �x���g�m�[�}��Y���� (-1 to 1), AO�ŎՕ�����Ȃ��������C�̐�) </returns>
        private (Color color, float bentNormalY, int unoccludedRays) CalculatePixelColor(
            Vector3 worldPos, Vector3 worldNormal, Material material,
            Vector2 lightmapUV, // ���C�g�}�b�v�e�N�Z���ɑΉ�����UV (�}�e���A���T���v�����O�Ƃ͕ʂ̏ꍇ����)
            Vector2 materialSampleUV // �}�e���A���v���p�e�B�T���v�����O�pUV (�W�b�^�����O����Ă���\������)
            )
        {
            MaterialProperties matProps = GetMaterialProperties(material);
            Color albedo = matProps.GetSampledBaseColor(materialSampleUV); // �W�b�^�����O���ꂽUV�Ń}�e���A�����T���v�����O
            var (metallic, roughness) = matProps.GetSampledMetallicRoughness(materialSampleUV);

            Color finalColor = Color.black;

            // 1. �X�J�C���C�g�v�Z
            Color skyContribution = CalculateSkyLight(worldPos, worldNormal, albedo);
            finalColor += skyContribution;

            // 2. ���ڌ� (�_�A�X�|�b�g�A�f�B���N�V���i��)
            foreach (Light light in sceneLights)
            {
                // (CalculatePBRDirectLight�Ăяo�������͕ύX�Ȃ��A�V���h�E�v�Z���C��)
                Vector3 lightDir;
                Color lightColorAtPoint = light.color * light.intensity;
                float attenuation = 1.0f;

                if (light.type == LightType.Directional) { /* ... */ lightDir = -light.transform.forward; }
                else
                {
                    Vector3 lightToPointVec = worldPos - light.transform.position;
                    float distSqr = lightToPointVec.sqrMagnitude;
                    if (distSqr > light.range * light.range) continue;
                    float dist = Mathf.Sqrt(distSqr);
                    lightDir = -lightToPointVec.normalized;
                    attenuation = 1.0f / (1.0f + 25.0f * distSqr / (light.range * light.range));
                    if (light.type == LightType.Spot)
                    {
                        float spotAngle = Vector3.Angle(light.transform.forward, -lightDir);
                        if (spotAngle > light.spotAngle / 2f) continue;
                        float outerConeRad = (light.spotAngle / 2f) * Mathf.Deg2Rad;
                        float cosOuter = Mathf.Cos(outerConeRad);
                        float cosAngle = Vector3.Dot(light.transform.forward, -lightDir);
                        float spotFalloff = Mathf.Pow(Mathf.Clamp01((cosAngle - cosOuter) / (1f - cosOuter + 1e-4f)), 2f);
                        attenuation *= spotFalloff;
                    }
                }

                // �\�t�g�V���h�E�v�Z
                float shadowAccumulator = 0f;
                int actualShadowSamples = Mathf.Max(1, settings.shadowSamples); // �Œ�1�T���v��
                for (int i = 0; i < actualShadowSamples; i++)
                {
                    Vector3 jitteredLightDir = lightDir;
                    Vector3 shadowRayOrigin = worldPos + worldNormal * 0.001f; // ���ȎՕ����

                    if (actualShadowSamples > 1)
                    {
                        // ���C�g�̎�ނƌ`��ɉ����ăW�b�^�����O���@��ς���̂����z
                        // �����ł͊ȈՓI�Ƀ��C�̕������킸���Ƀ����_����
                        // (�����T�C�Y���Ȃ����߁A�����I�ȃ\�t�g�V���h�E�Ƃ͈قȂ�ߎ�)
                        Vector3 randomOffset = Random.onUnitSphere * 0.05f; // �W�b�^�[�̋����i�v�����j
                        jitteredLightDir = (lightDir + randomOffset).normalized;
                    }

                    Ray shadowRay = new Ray(shadowRayOrigin, -jitteredLightDir); // ����������
                    float maxShadowDist = (light.type == LightType.Directional) ? 2000f : Vector3.Distance(worldPos, light.transform.position);
                    if (!Physics.Raycast(shadowRay, maxShadowDist - 0.002f)) // �q�b�g���Ȃ���Ό����͂�
                    {
                        shadowAccumulator += 1.0f;
                    }
                }
                float shadowFactor = shadowAccumulator / actualShadowSamples;

                if (shadowFactor > 0)
                {
                    Vector3 viewDir = worldNormal; // ���C�g�}�b�s���O�ł͖@���������Ƃ��邱�Ƃ�����
                    Color directLight = CalculatePBRDirectLight(
                        lightDir, viewDir, worldNormal, // �I���W�i����lightDir���g�p
                        lightColorAtPoint * attenuation * shadowFactor,
                        albedo, metallic, roughness
                    );
                    finalColor += directLight;
                }
            }

            // 3. �����ʂ���̊�^
            foreach (NeuraBakeEmissiveSurface emissiveSurface in emissiveSurfaces)
            {
                // �����ʂ���̃��C�e�B���O���v�Z (�ȈՓI�ȑ�����VPL�Ƃ��Ĉ���)
                // TODO: �����ʏ�̓_���T���v�����O���A�e�_����̊�^��ώZ
                // MeshFilter��Renderer���擾
                MeshFilter emissiveMeshFilter = emissiveSurface.GetComponent<MeshFilter>();
                Renderer emissiveRenderer = emissiveSurface.GetComponent<Renderer>();
                if (emissiveMeshFilter == null || emissiveMeshFilter.sharedMesh == null || emissiveRenderer == null) continue;

                Mesh emissiveMesh = emissiveMeshFilter.sharedMesh;
                int emissiveSamplePoints = 16; // 1�����ʂ�����̃T���v���_ (�ݒ�\�ɂ��Ă��ǂ�)

                for (int i = 0; i < emissiveSamplePoints; i++)
                {
                    // ���b�V���\�ʂ̃����_���ȓ_���擾 (�ʐϋϓ��T���v�����O�����z)
                    int randomTriangleIndex = Random.Range(0, emissiveMesh.triangles.Length / 3);
                    Vector3 b = GetRandomBarycentricCoords();

                    int vIdx0 = emissiveMesh.triangles[randomTriangleIndex * 3 + 0];
                    int vIdx1 = emissiveMesh.triangles[randomTriangleIndex * 3 + 1];
                    int vIdx2 = emissiveMesh.triangles[randomTriangleIndex * 3 + 2];

                    Vector3 emissivePointLocal = InterpolateVector3(emissiveMesh.vertices[vIdx0], emissiveMesh.vertices[vIdx1], emissiveMesh.vertices[vIdx2], b);
                    Vector3 emissiveNormalLocal = InterpolateVector3(emissiveMesh.normals[vIdx0], emissiveMesh.normals[vIdx1], emissiveMesh.normals[vIdx2], b).normalized;

                    Vector3 emissivePointWorld = emissiveSurface.transform.TransformPoint(emissivePointLocal);
                    Vector3 emissiveNormalWorld = emissiveSurface.transform.TransformDirection(emissiveNormalLocal).normalized;

                    Vector3 dirToEmissive = emissivePointWorld - worldPos;
                    float distToEmissiveSqr = dirToEmissive.sqrMagnitude;
                    float distToEmissive = Mathf.Sqrt(distToEmissiveSqr);
                    Vector3 normalizedDirToEmissive = dirToEmissive / distToEmissive;

                    // �����ʂ�������������Ă��邩�A������ʂ������ʂ������Ă��邩
                    float NdotL_emissive = Mathf.Max(0, Vector3.Dot(worldNormal, normalizedDirToEmissive));
                    float NdotL_source = Mathf.Max(0, Vector3.Dot(emissiveNormalWorld, -normalizedDirToEmissive)); // �����ʖ@���ƌ����ւ̋t�x�N�g��

                    if (NdotL_emissive > 0 && NdotL_source > 0)
                    {
                        // �����`�F�b�N
                        Ray visibilityRay = new Ray(worldPos + worldNormal * 0.001f, normalizedDirToEmissive);
                        if (!Physics.Raycast(visibilityRay, distToEmissive - 0.002f))
                        {
                            Color emissiveLightColor = emissiveSurface.emissiveColor * emissiveSurface.intensity * settings.emissiveBoost;
                            // ���� (1/dist^2) �ƃt�H�[���t�@�N�^�ߎ� (NdotL_emissive * NdotL_source / dist^2)
                            float formFactorApprox = (NdotL_emissive * NdotL_source) / (Mathf.PI * distToEmissiveSqr + 1e-4f); // PI�͖ʐς̍l��
                            formFactorApprox /= emissiveSamplePoints; // �T���v�����Ŋ���

                            // Diffuse BRDF�Ŏ��
                            Color contribution = (albedo / Mathf.PI) * emissiveLightColor * formFactorApprox;
                            finalColor += contribution;
                        }
                    }
                }
            }

            // 4. AO �� �x���g�m�[�}���v�Z
            float occlusionFactor = 0f;
            float bentNormalYComponent = 0f;
            int unoccludedRayCount = 0;

            if (settings.useAmbientOcclusion && settings.aoSampleCount > 0)
            {
                (occlusionFactor, bentNormalYComponent, unoccludedRayCount) = CalculateAmbientOcclusionAndBentNormal(worldPos, worldNormal, settings.aoSampleCount);
                finalColor *= (1f - occlusionFactor);
            }
            else if (settings.directional) // AO�����ł��x���g�m�[�}�����K�v�ȏꍇ (��̕���)
            {
                bentNormalYComponent = worldNormal.y; // �P���ɖ@����Y����
                unoccludedRayCount = 1; // �_�~�[�J�E���g
            }


            return (finalColor, bentNormalYComponent, unoccludedRayCount);
        }

        private Vector3 GetRandomBarycentricCoords()
        {
            float r1 = Random.value;
            float r2 = Random.value;
            float sqrtR1 = Mathf.Sqrt(r1);
            return new Vector3(1.0f - sqrtR1, sqrtR1 * (1.0f - r2), sqrtR1 * r2);
        }


        private Color CalculateSkyLight(Vector3 worldPos, Vector3 worldNormal, Color albedo)
        {
            if (RenderSettings.ambientMode == UnityEngine.Rendering.AmbientMode.Skybox && RenderSettings.skybox != null)
            {
                // Skybox�}�e���A������̃T���v�����O�͕��G�Ȃ̂ŁA�����ł͋ߎ��ɗ��߂�
                // SphericalHarmonicsL2 shEnv = RenderSettings.ambientProbe; // ����𒼐ڎg���͓̂��
                // Cubemap skyCubemap = RenderSettings.customReflection as Cubemap; // ������Ȃ��ꍇ������
                // �ł��P���ɂ� ambientSkyColor ���g�����A���ǂ�����ɂ͖@���ɉ����ăg���C���C�g���u�����h
            }

            Color skyColor = RenderSettings.ambientSkyColor; // �f�t�H���g
            if (RenderSettings.ambientMode == UnityEngine.Rendering.AmbientMode.Trilight)
            {
                float dotUp = Vector3.Dot(worldNormal, Vector3.up); // 0 (����) to 1 (�^��) or -1 (�^��)
                float upLerp = Mathf.Clamp01(dotUp);
                float downLerp = Mathf.Clamp01(-dotUp);
                skyColor = Color.Lerp(RenderSettings.ambientEquatorColor, RenderSettings.ambientSkyColor, upLerp);
                skyColor = Color.Lerp(skyColor, RenderSettings.ambientGroundColor, downLerp);
            }
            else if (RenderSettings.ambientMode == UnityEngine.Rendering.AmbientMode.Flat) // Flat �������� Skybox �ŏڍ׏��Ȃ�
            {
                skyColor = RenderSettings.ambientLight; // Flat�̏ꍇ
            }
            // Skybox �̏ꍇ�A�{���� RenderSettings.ambientProbe (SH) ��]�����邩�A
            // Skybox Cubemap ���T���v�����O����̂��]�܂����B
            // �����ł͒P�������āA�������A���x�h�ŕϒ��B
            return skyColor * albedo * settings.skyIntensity * RenderSettings.ambientIntensity;
        }

        // PBR Direct Light �v�Z (�O�񂩂�ύX�Ȃ�)
        private Color CalculatePBRDirectLight(Vector3 L, Vector3 V, Vector3 N, Color lightColor, Color albedo, float metallic, float roughness)
        {
            L = L.normalized; V = V.normalized; N = N.normalized;
            Vector3 H = (L + V).normalized;
            float NdotL = Mathf.Max(0f, Vector3.Dot(N, L));
            if (NdotL <= 0f) return Color.black;
            float NdotV = Mathf.Max(0f, Vector3.Dot(N, V));
            float alpha = roughness * roughness;
            Color F0 = Color.Lerp(new Color(0.04f, 0.04f, 0.04f), albedo, metallic);
            float D_GGX = GGX_Distribution(N, H, alpha);
            Color F_Schlick = Fresnel_Schlick(Mathf.Max(0f, Vector3.Dot(H, V)), F0);
            float G_SmithJointGGX = Smith_Visibility_JointGGX(N, V, L, alpha);
            Color specular = (D_GGX * F_Schlick * G_SmithJointGGX) / (4f * NdotL * NdotV + 1e-6f);
            Color kd = Color.Lerp(Color.one, Color.black, metallic);
            Color diffuse = kd * albedo / Mathf.PI;
            return (diffuse + specular) * lightColor * NdotL;
        }
        private float GGX_Distribution(Vector3 N, Vector3 H, float alpha) { /* ... */ float NdotH = Mathf.Max(0f, Vector3.Dot(N, H)); float alphaSq = alpha * alpha; float denom = NdotH * NdotH * (alphaSq - 1f) + 1f; return alphaSq / (Mathf.PI * denom * denom); }
        private Color Fresnel_Schlick(float cosTheta, Color F0) { /* ... */ return F0 + (new Color(1f - F0.r, 1f - F0.g, 1f - F0.b)) * Mathf.Pow(1f - cosTheta, 5f); } // (1-F0)�̌v�Z���C��
        private float Smith_Visibility_JointGGX(Vector3 N, Vector3 V, Vector3 L, float alpha) { /* ... */ float NdotV = Mathf.Max(0f, Vector3.Dot(N, V)); float NdotL = Mathf.Max(0f, Vector3.Dot(N, L)); float k_direct = (alpha + 2f * Mathf.Sqrt(alpha) + 1f) / 8f; float G_V = NdotV / (NdotV * (1f - k_direct) + k_direct + 1e-5f); float G_L = NdotL / (NdotL * (1f - k_direct) + k_direct + 1e-5f); return G_V * G_L; }


        /// <summary>
        /// AO�ƃx���g�m�[�}��Y���v�Z���܂��B
        /// </summary>
        /// <returns>(occlusionFactor, bentNormal.y, unoccludedRayCount)</returns>
        private (float occlusion, float bentNormalY, int unoccludedRays) CalculateAmbientOcclusionAndBentNormal(Vector3 position, Vector3 normal, int sampleCount)
        {
            if (sampleCount <= 0) return (0f, normal.y, 0); // AO�Ȃ��Ȃ�@����Y��Ԃ�

            float occlusionFactor = 0f;
            Vector3 accumulatedUnoccludedNormal = Vector3.zero;
            int unoccludedRayCount = 0;
            float maxAoDistance = 2.0f; // AO�̗L�����a

            for (int i = 0; i < sampleCount; i++)
            {
                Vector3 randomDir = Random.onUnitSphere;
                if (Vector3.Dot(randomDir, normal) < 0)
                {
                    randomDir = -randomDir;
                }

                Ray aoRay = new Ray(position + normal * 0.001f, randomDir);
                if (!Physics.Raycast(aoRay, maxAoDistance)) // �q�b�g���Ȃ���ΎՕ�����Ă��Ȃ�
                {
                    accumulatedUnoccludedNormal += randomDir;
                    unoccludedRayCount++;
                }
                else // �q�b�g������Օ�
                {
                    occlusionFactor += 1f;
                }
            }

            float finalOcclusion = (sampleCount > 0) ? occlusionFactor / sampleCount : 0f;
            float finalBentNormalY = normal.y; // �f�t�H���g�͌��̖@��Y

            if (unoccludedRayCount > 0)
            {
                finalBentNormalY = (accumulatedUnoccludedNormal / unoccludedRayCount).normalized.y;
            }

            return (finalOcclusion, finalBentNormalY, unoccludedRayCount);
        }
    }
}